#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Keyboard controlling for CARLA. Please refer to client_example.py for a simpler
# and more documented example.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    R            : restart level

STARTING in a moment...
"""


import argparse
import cv2
import logging
import random
import time
import math
import colorsys
import os
import sys
import glob
from queue import Queue
from constants import *
from queue import Empty

sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' %
                          (sys.version_info.major, sys.version_info.minor, 'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])

try:
    import pygame
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
    from numpy.linalg import pinv, inv
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

import carla
from utils import Timer, rand_color, vector3d_to_array, degrees_to_radians
from datadescriptor import KittiDescriptor
from dataexport import *
from carla_utils import KeyboardHelper, MeasurementsDisplayHelper
from constants import *
import lidar_utils  # from lidar_utils import project_point_cloud
import time
from math import cos, sin

from carla import ColorConverter as cc

""" OUTPUT FOLDER GENERATION """
PHASE = "training"
OUTPUT_FOLDER = "/media/oem/Local Disk/Phd-datasets/Carla_dataset/sequences/00"
# folders = ['calib', 'image_2', 'label_2', 'velodyne', 'planes']


# def maybe_create_dir(path):
#     if not os.path.exists(directory):
#         os.makedirs(directory)


# for folder in folders:
#     directory = os.path.join(OUTPUT_FOLDER, folder)
#     maybe_create_dir(directory)

""" DATA SAVE PATHS """
LIDAR_PATH = os.path.join(OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
LABEL_PATH = os.path.join(OUTPUT_FOLDER, 'labels/{0:06}.label')
IMAGE_PATH = os.path.join(OUTPUT_FOLDER, 'images/{0:06}.png')
CALIBRATION_PATH = os.path.join(OUTPUT_FOLDER, 'calib/calib.txt')


class CarlaGame(object):
    def __init__(self, carla_client, args):
        self.client = carla_client
        self.client.set_timeout(10.0)
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_global_distance_to_leading_vehicle(0.5)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_hybrid_physics_mode(True)
        self.just_destroyed = False
        self.camera = None
        self.camera_seg = None
        self.lidar = None
        self._timer = None
        self._display = None
        self._main_image = None
        self._mini_view_image1 = None
        self._mini_view_image2 = None
        self._enable_autopilot = AUTO_PILOT
        self._lidar_measurement = None
        self._is_on_reverse = False
        self._city_name = args.map_name
        self._position = None
        self._agent_positions = None
        self.captured_frame_no = self.current_captured_frame_num()
        self._extrinsic = None
        # To keep track of how far the car has driven since the last capture of data
        self._agent_location_on_last_capture = None
        # How many frames we have captured since reset
        self._captured_frames_since_restart = 0
        self.player = None
        self.walker_actors = []
        self.walker_controller_list = []
        self.vehicles_actors = []
        lidar_points_per_sec = LIDAR_NUM_CHANNELS * 2048 * LIDAR_FPS
        self.sensors = [
            ['sensor.camera.rgb', {'image_size_x': str(WINDOW_WIDTH), 'image_size_y': str(WINDOW_HEIGHT)}],
            ['sensor.camera.semantic_segmentation', {'image_size_x': str(WINDOW_WIDTH), 'image_size_y': str(WINDOW_HEIGHT)}],
            ['sensor.lidar.ray_cast', {'range': str(LIDAR_MAX_RANGE), 'points_per_second': str(lidar_points_per_sec), 'channels': str(LIDAR_NUM_CHANNELS),
                                                                 'lower_fov': '-25.0', 'upper_fov': '3.0', 'dropoff_general_rate': '0.0', 'dropoff_intensity_limit': '0.5',
                                                                  'rotation_frequency' : str(LIDAR_FPS)}]
            ]
        """Make a CarlaSettings object with the settings we need."""


    def current_captured_frame_num(self):
        # Figures out which frame number we currently are on
        # This is run once, when we start the simulator in case we already have a dataset.
        # The user can then choose to overwrite or append to the dataset.
        label_path = os.path.join(OUTPUT_FOLDER, 'velodyne/')
        num_existing_data_files = len(
            [name for name in os.listdir(label_path) if name.endswith('.bin')])
        print(num_existing_data_files)
        if num_existing_data_files == 0:
            return 0
        answer = input(
            "There already exists a dataset in {}. Would you like to (O)verwrite or (A)ppend the dataset? (O/A)".format(OUTPUT_FOLDER))
        if answer.upper() == "O":
            logging.info(
                "Resetting frame number to 0 and overwriting existing")
            # Overwrite the data
            return 0
        logging.info("Continuing recording data on frame number {}".format(
            num_existing_data_files))
        return num_existing_data_files

    def execute(self):
        """Launch the PyGame."""
        pygame.init()
        self._initialize_game()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                reset = self._on_loop()
                if not reset:
                    self._on_render()
        finally:
            pygame.quit()
            self.world.apply_settings(self.original_settings)
            self.destroy()


    def _initialize_game(self):

        self._display = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        logging.debug('pygame started')
        self._on_new_episode()


    def spawn(self, blueprint, transform=None, attach_to=None):
        if transform is None:
            actor = None
            while True:
                if not self.map.get_spawn_points():
                    logging.error('no spawning points')
                    exit(1)
                spawn_points = self.map.get_spawn_points()
                spawn_point = random.choice(spawn_points)
                actor = self.world.try_spawn_actor(blueprint, spawn_point)
                if actor is not None:
                    return actor
        elif attach_to is None:
            actor = self.world.try_spawn_actor(blueprint=blueprint, transform=transform)
            return actor
        actor = self.world.try_spawn_actor(blueprint=blueprint, transform=transform, attach_to=attach_to)
        return actor


    def _on_new_episode(self):

        map_n = 2
        self.client.load_world('Town0'+ str(map_n))
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.no_rendering_mode = NO_RENDERING_MODE
        settings.fixed_delta_seconds = 1.0 / LIDAR_FPS
        self.map = self.world.get_map()
        self.world.apply_settings(settings)


        if self.player is not None:
            self.destroy()
        self.player = None    
        self.all_id = []
        self.all_actors = []
        self.vehicles_actors = []

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor


        blueprint = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # Spawn the player.
        self.player = self.spawn(blueprint)
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        n_vehicles = NUM_VEHICLES
        if len(spawn_points) < n_vehicles:
            n_vehicles = len(spawn_points)
        batch = []

        for i in range(n_vehicles):
            bp = random.choice(self.world.get_blueprint_library().filter('vehicle.*'))
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            batch.append(SpawnActor(bp, spawn_points[i])
                         .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port())))
        
        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_actors.append(response.actor_id)

        # Walkers ....
        spawn_points = []
        for i in range(NUM_PEDESTRIANS):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        walker_speed = []    
        percentagePedestriansRunning = 0.0
        percentagePedestriansCrossing = 0.3
        batch = []
        for spawn_point in spawn_points:
            bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(bp, spawn_point))

            # self.walker_actors.append(actor)

        walkers_list=[]
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2


        batch = []
        for i in range(len(walkers_list)):
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
            
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        self.all_id = []
        for i in range(len(walkers_list)):
            self.all_id.append(walkers_list[i]["con"])
            self.all_id.append(walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)

        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(
                self.world.get_random_location_from_navigation())
            # max speed
            self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        # Set up the sensors.
        self.traffic_manager.global_percentage_speed_difference(30.0)

        sensors_bp_dict = dict()
        bp_library = self.world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            sensors_bp_dict[item[0]] = bp
            for attr_name, attr_value in item[1].items():
                bp.set_attribute(attr_name, attr_value)
            
        self.camera = self.spawn(sensors_bp_dict['sensor.camera.rgb'], carla.Transform(
            carla.Location(x=1.6, z=1.6)), attach_to=self.player)
        self.camera_seg = self.spawn(sensors_bp_dict['sensor.camera.semantic_segmentation'], carla.Transform(
            carla.Location(x=1.6, z=1.6)), attach_to=self.player)
        self.lidar = self.spawn(sensors_bp_dict['sensor.lidar.ray_cast'], carla.Transform(
            carla.Location(x=1.0, z=1.8)), attach_to=self.player)

        camera_fov = sensors_bp_dict['sensor.camera.rgb'].get_attribute('fov').as_float()
        camera_width = sensors_bp_dict['sensor.camera.rgb'].get_attribute('image_size_x').as_int()
        camera_height = sensors_bp_dict['sensor.camera.rgb'].get_attribute('image_size_y').as_int()

        focal = camera_width / (2.0 * np.tan(camera_fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = camera_width / 2.0
        K[1, 2] = camera_height / 2.0
        self.intrinsic = K
        
    

        logging.info('Starting new episode...')
        self._timer = Timer()
        self._is_on_reverse = False

        self.image_queue = Queue()
        self.lidar_queue = Queue()
        self.camera_seg_queue = Queue()

        self.camera.listen(lambda data: self.image_queue.put(data))
        self.lidar.listen(lambda data: self.lidar_queue.put(data))
        self.camera_seg.listen(lambda data: self.camera_seg_queue.put(data))

        # Reset all tracking variables
        self._agent_location_on_last_capture = None
        self._captured_frames_since_restart = 0
        self.just_destroyed = False

    def get_rgb_array_from_carla_image(self, carla_image):
        im_array = np.copy(np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8")))
        im_array = np.reshape(im_array, (carla_image.height, carla_image.width, 4))
        im_array = im_array[:, :, :3][:, :, ::-1]
        return im_array

    def lidar_to_camera_trans(self, pc, l_2_w_trans, w_2_c_trans,  K, image_w, image_h):
        pc = np.r_[pc, [np.ones(pc.shape[1])]]

        # Transform the points from lidar space to world space.
        world_points = np.dot(l_2_w_trans, pc)

        # This (4, 4) matrix transforms the points from world to sensor coordinates.

        # Transform the points from world space to camera space.
        sensor_points = np.dot(w_2_c_trans, world_points)
        point_in_camera_coords = np.array([
            sensor_points[1],
            sensor_points[2] * -1,
            sensor_points[0]])

        # Finally we can use our K matrix to do the actual 3D -> 2D.
        points_2d = np.dot(K, point_in_camera_coords)
        # Remember to normalize the x, y values by the 3rd value.
        points_2d = np.array([
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :]])

        # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
        # contains all the y values of our points. In order to properly
        # visualize everything on a screen, the points that are out of the screen
        # must be discarted, the same with points behind the camera projection plane.
        points_2d = points_2d.T
        points_in_canvas_mask = \
            (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
            (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
            (points_2d[:, 2] > 0.0)
        points_2d = points_2d[points_in_canvas_mask]

        # Extract the screen coords (uv) as integers.
        u_coord = points_2d[:, 0].astype(np.int)
        v_coord = points_2d[:, 1].astype(np.int)
        return u_coord, v_coord, points_in_canvas_mask


    def _on_loop(self):
       
        # Reset the environment if the agent is stuck or can't find any agents or if we have captured enough frames in this one
        is_enough_datapoints = (self._captured_frames_since_restart + 1) % NUM_RECORDINGS_BEFORE_RESET == 0

        if (is_enough_datapoints) and GEN_DATA:
            logging.warning("Is_enough_datapoints: {}".format(is_enough_datapoints))
            self._on_new_episode()
            # If we dont sleep, the client will continue to render
            return True
        # if self._timer.step > 0:
            # print(f'Time elapsed between loops :{self._timer.elapsed_seconds_since_lap()}\n')
        self._timer.lap()
        self._timer.tick()
        self.world.tick()

        world_frame = self.world.get_snapshot().frame
        while True:
            try:
                # Get the data once it's received.
                image_data = self.image_queue.get(True, 1.0)
                lidar_data = self.lidar_queue.get(True, 1.0)
                camera_seg_data = self.camera_seg_queue.get(True, 1.0)
                break
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue
            
        assert image_data.frame == lidar_data.frame == world_frame

        self._last_player_location = self.player.get_location()
        self.image_data = image_data
        self.lidar_data = lidar_data
        self.camera_seg_data = camera_seg_data
        # Get the raw BGRA buffer and convert it to an array of RGB of
        # shape (image_data.height, image_data.width, 3).
        im_array = self.get_rgb_array_from_carla_image(image_data)
        im_seg_array = self.get_rgb_array_from_carla_image(camera_seg_data)
        p_cloud_size = len(lidar_data)
        p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
        p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))
        local_lidar_points = np.array(p_cloud[:, :3]).T
        self.vel_2_ref = self.lidar.get_transform().get_matrix()
        self.extrinsic = self.camera.get_transform().get_inverse_matrix()
        u_coord, v_coord, mask = self.lidar_to_camera_trans(local_lidar_points, self.vel_2_ref, self.extrinsic, self.intrinsic, image_data.width, image_data.height)
        color_map = LABEL_COLORS[im_seg_array[:, :, 0][v_coord, u_coord]]
        dot_extent = 1
        p_cloud_rgb = np.copy(im_array[v_coord, u_coord,:])
        im_array[v_coord, u_coord] = color_map
        # for i in range(len(u_coord)):
        #         im_array[
        #             v_coord[i]-dot_extent : v_coord[i]+dot_extent,
        #             u_coord[i]-dot_extent : u_coord[i]+dot_extent] = color_map[i]

        self._main_image = im_array
        self.p_cloud_labels = CARLA_2_KITTI_LABEL_MAP[im_seg_array[:, :, 0][v_coord, u_coord]].astype('int32')
        self.p_cloud = np.concatenate((p_cloud[mask], p_cloud_rgb.astype(np.float32)), axis=-1)
        control = self._get_keyboard_control(pygame.key.get_pressed())
        if control is None:
            self._on_new_episode()
        else:
            self.player.apply_control(control)

        if self._enable_autopilot:
            self.player.set_autopilot(True, self.traffic_manager.get_port())
        else:
            self.player.set_autopilot(False)



    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        control = KeyboardHelper.get_keyboard_control(
            keys, self._is_on_reverse, self._enable_autopilot)
        if control is not None:
            control, self._is_on_reverse, self._enable_autopilot = control
        return control

    def _on_render(self):

        if self._main_image is not None:
            # Convert main image

            # Retrieve and draw datapoints
            # Display image
            surface = pygame.surfarray.make_surface(self._main_image.copy().swapaxes(0, 1))
            self._display.blit(surface, (0, 0))
            pygame.display.flip()

            # Determine whether to save files
            distance_driven = self._distance_since_last_recording()
            #print("Distance driven since last recording: {}".format(distance_driven))
            has_driven_long_enough = distance_driven is None or distance_driven > DISTANCE_SINCE_LAST_RECORDING
            if (self._timer.step + 1) % STEPS_BETWEEN_RECORDINGS == 0:
                if has_driven_long_enough:
                    # Save screen, lidar and kitti training labels together with calibration and groundplane files
                    self._update_agent_location()
                    self._save_training_files()
                    self.captured_frame_no += 1
                    self._captured_frames_since_restart += 1
                else:
                    logging.debug("Could save datapoint, but agent has not driven {} meters since last recording (Currently {} meters)".format(
                        DISTANCE_SINCE_LAST_RECORDING, distance_driven))

    def _distance_since_last_recording(self):
        if self._agent_location_on_last_capture is None:
            return None
        cur_pos = vector3d_to_array(self.player.get_location())
        last_pos = vector3d_to_array(self._agent_location_on_last_capture)
        def dist_func(x, y): return np.sqrt(((x - y)**2).sum())

        return dist_func(cur_pos, last_pos)

    def _update_agent_location(self):
        self._agent_location_on_last_capture = self.player.get_location()
        
    def _save_training_files(self):
        # self.extrinsic = np.dot(unreal_to_world_cam, self.camera.get_transform().get_inverse_matrix())
        # self.vel_2_ref = np.dot(self.lidar.get_transform(), unreal_to_world_3d)
        logging.info("Attempting to save at timer step {}, frame no: {}".format(
            self._timer.step, self.captured_frame_no))
        lidar_fname = LIDAR_PATH.format(self.captured_frame_no)
        label_fname = LABEL_PATH.format(self.captured_frame_no)
        img_fname = IMAGE_PATH.format(self.captured_frame_no)
        calib_filename = CALIBRATION_PATH

        # save_ref_files(OUTPUT_FOLDER, self.captured_frame_no)
        save_image_data(img_fname, self._main_image)
        save_lidar_data(lidar_fname, self.p_cloud)
        save_lidar_labels(label_fname, self.p_cloud_labels)
        save_calibration_matrices(calib_filename, self.intrinsic, self.extrinsic, self.vel_2_ref)

    def destroy(self):
        if not self.just_destroyed:
            sensors = [
                self.camera,
                self.camera_seg,
                self.lidar]
            for sensor in sensors:
                if sensor is not None:
                    sensor.stop()
                    sensor.destroy()
            if self.player is not None:
                self.player.destroy()

            for i in range(0, len(self.all_id), 2):
                self.all_actors[i].stop()
            vehicle_and_walkers = []
            vehicle_and_walkers.extend(self.vehicles_actors)
            vehicle_and_walkers.extend(self.all_id)
            self.client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_and_walkers])
            print('\ndestroying %d actors' % int(len(self.all_id)//2 + len(self.vehicles_actors)))
            self.just_destroyed = True
            time.sleep(0.5)
        else:
            print('destroy has been just called\n')



def parse_args():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='logging.info debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-m', '--map-name',
        metavar='M',
        default=None,
        help='plot the map of the current city (needs to match active map in '
             'server, options: Town01 or Town02)')
    args = argparser.parse_args()
    return args


def main():
    args = parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    logging.info(__doc__)
    game = None
    try:
        client = carla.Client(args.host, args.port)
        game = CarlaGame(client, args)
        game.execute()
    except ConnectionError as error:
        logging.error(error)
    except KeyboardInterrupt:
        if game is not None:
            game.destroy()
        logging.info('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
