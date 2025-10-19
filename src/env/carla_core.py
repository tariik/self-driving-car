#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
import random
import signal
import subprocess
import time
import psutil
import logging

import carla

from src.env.helper import join_dicts
from src.env.sensors.factory import SensorFactory
from src.env.sensors.sensor_interface import SensorInterface


BASE_CORE_CONFIG = {
    "host": "localhost",  # Client host
    "port": 2000,  # Port of the CARLA server (if using external server)
    "timeout": 10.0,  # Timeout of the client
    "timestep": 0.05,  # Time step of the simulation
    "retries_on_error": 10,  # Number of tries to connect to the client
    "resolution_x": 600,  # Width of the server spectator camera
    "resolution_y": 600,  # Height of the server spectator camera
    "quality_level": "Low",  # Quality level of the simulation. Can be 'Low', 'High', 'Epic'
    "enable_map_assets": False,  # enable / disable all town assets except for the road
    "enable_rendering": True,  # enable / disable camera images
    "show_display": False,  # Whether or not the server will be displayed
    "use_external_server": True  # Whether to use an external CARLA server or start a new one
}


def is_used(port):
    """Checks whether or not a port is used"""
    return port in [conn.laddr.port for conn in psutil.net_connections()]

def kill_all_servers():
    """Kill all PIDs that start with Carla"""
    processes = [p for p in psutil.process_iter() if "carla" in p.name().lower()]
    for process in processes:
        os.kill(process.pid, signal.SIGKILL)


class CarlaCore:
    """
    Class responsible of handling all the different CARLA functionalities, such as server-client connecting,
    actor spawning and getting the sensors data.
    """
    def __init__(self, config={}):
        """Initialize the server and client"""
        self.client = None
        self.world = None
        self.map = None
        self.hero = None
        self.config = join_dicts(BASE_CORE_CONFIG, config)
        self.sensor_interface = SensorInterface()

        # Only start server if not using an external one
        if not self.config.get("use_external_server", False):
            self.init_server()
        else:
            self.server_port = self.config.get("port", 2000)
        
        self.connect_client()

    def init_server(self):
        """Start a server on a random port"""
        self.server_port = random.randint(15000, 32000)

        # Ray tends to start all processes simultaneously. Use random delays to avoid problems
        time.sleep(random.uniform(0, 1))

        uses_server_port = is_used(self.server_port)
        uses_stream_port = is_used(self.server_port + 1)
        while uses_server_port and uses_stream_port:
            if uses_server_port:
                print("Is using the server port: " + self.server_port)
            if uses_stream_port:
                print("Is using the streaming port: " + str(self.server_port+1))
            self.server_port += 2
            uses_server_port = is_used(self.server_port)
            uses_stream_port = is_used(self.server_port+1)

        if self.config["show_display"]:
            server_command = [
                "{}/CarlaUE4.sh".format(os.environ["CARLA_ROOT"]),
                "-windowed",
                "-ResX={}".format(self.config["resolution_x"]),
                "-ResY={}".format(self.config["resolution_y"]),
            ]
        else:
            server_command = [
                "DISPLAY= ",
                "{}/CarlaUE4.sh".format(os.environ["CARLA_ROOT"]),
                "-opengl"  # no-display isn't supported for Unreal 4.24 with vulkan
            ]

        server_command += [
            "--carla-rpc-port={}".format(self.server_port),
            "-quality-level={}".format(self.config["quality_level"])
        ]

        server_command_text = " ".join(map(str, server_command))
        print(server_command_text)
        server_process = subprocess.Popen(
            server_command_text,
            shell=True,
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w"),
        )

    def connect_client(self):
        """Connect to the client"""

        for i in range(self.config["retries_on_error"]):
            try:
                self.client = carla.Client(self.config["host"], self.server_port)
                self.client.set_timeout(self.config["timeout"])
                self.world = self.client.get_world()

                settings = self.world.get_settings()
                settings.no_rendering_mode = not self.config["enable_rendering"]
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.config["timestep"]
                self.world.apply_settings(settings)
                self.world.tick()

                return

            except Exception as e:
                print(" Waiting for server to be ready: {}, attempt {} of {}".format(e, i + 1, self.config["retries_on_error"]))
                time.sleep(3)

        raise Exception("Cannot connect to server. Try increasing 'timeout' or 'retries_on_error' at the carla configuration")

    def setup_experiment(self, experiment_config):
        """Initialize the hero and sensors"""

        # Check if we need to load a different map
        current_map = self.world.get_map().name
        requested_map = experiment_config["town"]
        
        # Only load the world if the map is different
        if requested_map not in current_map:
            print(f"Loading map {requested_map}...")
            self.world = self.client.load_world(
                map_name = requested_map,
                reset_settings = False,
                map_layers = carla.MapLayer.All if self.config["enable_map_assets"] else carla.MapLayer.NONE)
        else:
            print(f"Map {requested_map} is already loaded, skipping reload...")

        self.map = self.world.get_map()

        # Choose the weather of the simulation
        weather = getattr(carla.WeatherParameters, experiment_config["weather"])
        self.world.set_weather(weather)

        # Clear all vehicles only if flag enabled (clean road)
        if experiment_config.get("clean_road", False):
            try:
                vehicles = list(self.world.get_actors().filter('vehicle.*'))
                removed = 0
                for v in vehicles:
                    # Remove every vehicle (hero will be respawned later)
                    try:
                        v.destroy()
                        removed += 1
                    except Exception:
                        pass
                if removed:
                    self.world.tick()
                    print(f"Removed {removed} vehicles from world (clean road)")
            except Exception as e:
                print(f"Warning while clearing vehicles: {e}")

        # Background NPCs / Traffic Manager configuration
        n_vehicles = experiment_config["background_activity"]["n_vehicles"]
        n_walkers = experiment_config["background_activity"]["n_walkers"]
        always_init_tm = experiment_config.get("always_init_traffic_manager", False)
        want_tm = always_init_tm or (n_vehicles > 0 or n_walkers > 0)

        if not want_tm:
            # Do not initialize Traffic Manager and do not spawn background actors
            self.traffic_manager = None
            print("Traffic Manager disabled (no NPCs and not forced); skipping background spawning")
        else:
            # Initialize Traffic Manager
            self.tm_port = self.server_port // 10 + self.server_port % 10
            while is_used(self.tm_port):
                print("Traffic manager's port " + str(self.tm_port) + " is already being used. Checking the next one")
                self.tm_port += 1
            print("Traffic manager connected to port " + str(self.tm_port))

            try:
                self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
                self.traffic_manager.set_hybrid_physics_mode(experiment_config["background_activity"]["tm_hybrid_mode"])
                seed = experiment_config["background_activity"]["seed"]
                if seed is not None:
                    self.traffic_manager.set_random_device_seed(seed)
            except RuntimeError as e:
                print(f"Warning: Could not create traffic manager on port {self.tm_port}: {e}")
                print("Attempting to use default traffic manager port 8000...")
                try:
                    self.traffic_manager = self.client.get_trafficmanager(8000)
                    self.traffic_manager.set_hybrid_physics_mode(experiment_config["background_activity"]["tm_hybrid_mode"])
                    seed = experiment_config["background_activity"]["seed"]
                    if seed is not None:
                        self.traffic_manager.set_random_device_seed(seed)
                except Exception as e2:
                    print(f"Warning: Could not connect to traffic manager: {e2}")
                    print("Continuing without traffic manager. Background traffic will not be available.")
                    self.traffic_manager = None

            # Spawn the background activity only if requested
            if n_vehicles > 0 or n_walkers > 0:
                self.spawn_npcs(n_vehicles, n_walkers)
            else:
                print("TM initialized (forced), but no NPCs requested; not spawning any background actors")


    def reset_hero(self, hero_config):
        """This function resets / spawns the hero vehicle and its sensors"""

        # Part 1: destroy all sensors (if necessary)
        self.sensor_interface.destroy()

        self.world.tick()

        # Part 2: Spawn the ego vehicle
        user_spawn_points = hero_config["spawn_points"]
        if user_spawn_points:
            spawn_points = []
            for transform in user_spawn_points:

                transform = [float(x) for x in transform.split(",")]
                if len(transform) == 3:
                    location = carla.Location(
                        transform[0], transform[1], transform[2]
                    )
                    waypoint = self.map.get_waypoint(location)
                    waypoint = waypoint.previous(random.uniform(0, 5))[0]
                    transform = carla.Transform(
                        location, waypoint.transform.rotation
                    )
                else:
                    assert len(transform) == 6
                    transform = carla.Transform(
                        carla.Location(transform[0], transform[1], transform[2]),
                        carla.Rotation(transform[4], transform[5], transform[3])
                    )
                spawn_points.append(transform)
        else:
            spawn_points = self.map.get_spawn_points()

        self.hero_blueprints = self.world.get_blueprint_library().find(hero_config['blueprint'])
        self.hero_blueprints.set_attribute("role_name", "hero")

        # If already spawned, destroy it
        if self.hero is not None:
            self.hero.destroy()
            self.hero = None

        random.shuffle(spawn_points)
        for i in range(0,len(spawn_points)):
            next_spawn_point = spawn_points[i % len(spawn_points)]
            self.hero = self.world.try_spawn_actor(self.hero_blueprints, next_spawn_point)
            if self.hero is not None:
                print("Hero spawned!")
                break
            else:
                print("Could not spawn hero, changing spawn point")

        if self.hero is None:
            print("We ran out of spawn points")
            return

        self.world.tick()

        # Part 3: Spawn the new sensors
        for name, attributes in hero_config["sensors"].items():
            sensor = SensorFactory.spawn(name, attributes, self.sensor_interface, self.hero)

        # Not needed anymore. This tick will happen when calling CarlaCore.tick()
        # self.world.tick()

        return self.hero

    def spawn_npcs(self, n_vehicles, n_walkers):
        """Spawns vehicles and walkers, also setting up the Traffic Manager and its parameters"""

        # Skip spawning NPCs if traffic manager is not available or no NPCs requested
        if self.traffic_manager is None:
            print("Traffic manager not available, skipping NPC spawning")
            return
            
        if n_vehicles == 0 and n_walkers == 0:
            print("No NPCs requested, skipping NPC spawning")
            return

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # Spawn vehicles
        spawn_points = self.world.get_map().get_spawn_points()
        n_spawn_points = len(spawn_points)

        if n_vehicles < n_spawn_points:
            random.shuffle(spawn_points)
        elif n_vehicles > n_spawn_points:
            logging.warning("{} vehicles were requested, but there were only {} available spawn points"
                            .format(n_vehicles, n_spawn_points))
            n_vehicles = n_spawn_points

        v_batch = []
        v_blueprints = self.world.get_blueprint_library().filter("vehicle.*")

        for n, transform in enumerate(spawn_points):
            if n >= n_vehicles:
                break
            v_blueprint = random.choice(v_blueprints)
            if v_blueprint.has_attribute('color'):
                color = random.choice(v_blueprint.get_attribute('color').recommended_values)
                v_blueprint.set_attribute('color', color)
            v_blueprint.set_attribute('role_name', 'autopilot')

            transform.location.z += 1
            v_batch.append(SpawnActor(v_blueprint, transform)
                           .then(SetAutopilot(FutureActor, True, self.tm_port)))

        results = self.client.apply_batch_sync(v_batch, True)
        if len(results) < n_vehicles:
            logging.warning("{} vehicles were requested but could only spawn {}"
                            .format(n_vehicles, len(results)))
        vehicles_id_list = [r.actor_id for r in results if not r.error]

        # Spawn the walkers
        spawn_locations = [self.world.get_random_location_from_navigation() for i in range(n_walkers)]

        w_batch = []
        w_blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")

        for spawn_location in spawn_locations:
            w_blueprint = random.choice(w_blueprints)
            if w_blueprint.has_attribute('is_invincible'):
                w_blueprint.set_attribute('is_invincible', 'false')
            w_batch.append(SpawnActor(w_blueprint, carla.Transform(spawn_location)))

        results = self.client.apply_batch_sync(w_batch, True)
        if len(results) < n_walkers:
            logging.warning("Could only spawn {} out of the {} requested walkers."
                            .format(len(results), n_walkers))
        walkers_id_list = [r.actor_id for r in results if not r.error]

        # Spawn the walker controllers
        wc_batch = []
        wc_blueprint = self.world.get_blueprint_library().find('controller.ai.walker')

        for walker_id in walkers_id_list:
            wc_batch.append(SpawnActor(wc_blueprint, carla.Transform(), walker_id))

        results = self.client.apply_batch_sync(wc_batch, True)
        if len(results) < len(walkers_id_list):
            logging.warning("Only {} out of {} controllers could be created. Some walkers might be stopped"
                            .format(len(results), n_walkers))
        controllers_id_list = [r.actor_id for r in results if not r.error]

        self.world.tick()

        for controller in self.world.get_actors(controllers_id_list):
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())

        self.world.tick()
        self.actors = self.world.get_actors(vehicles_id_list + walkers_id_list + controllers_id_list)

    def tick(self, control):
        """Performs one tick of the simulation, moving all actors, and getting the sensor data"""

        # Move hero vehicle
        if control is not None:
            self.apply_hero_control(control)

        # Tick once the simulation
        self.world.tick()

        # Move the spectator
        if self.config["enable_rendering"]:
            self.set_spectator_camera_view()

        # Return the new sensor data
        return self.get_sensor_data()

    def set_spectator_camera_view(self):
        """This positions the spectator as a 3rd person view of the hero vehicle"""
        transform = self.hero.get_transform()

        # Get the camera position
        server_view_x = transform.location.x - 5 * transform.get_forward_vector().x
        server_view_y = transform.location.y - 5 * transform.get_forward_vector().y
        server_view_z = transform.location.z + 3

        # Get the camera orientation
        server_view_roll = transform.rotation.roll
        server_view_yaw = transform.rotation.yaw
        server_view_pitch = transform.rotation.pitch

        # Get the spectator and place it on the desired position
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(
            carla.Transform(
                carla.Location(x=server_view_x, y=server_view_y, z=server_view_z),
                carla.Rotation(pitch=server_view_pitch,yaw=server_view_yaw,roll=server_view_roll),
            )
        )

    def apply_hero_control(self, control):
        """Applies the control calcualted at the experiment to the hero"""
        self.hero.apply_control(control)

    def get_sensor_data(self):
        """Returns the data sent by the different sensors at this tick"""
        sensor_data = self.sensor_interface.get_data()
        # print("---------")
        # world_frame = self.world.get_snapshot().frame
        # print("World frame: {}".format(world_frame))
        # for name, data in sensor_data.items():
        #     print("{}: {}".format(name, data[0]))
        return sensor_data