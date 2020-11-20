import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np

from planning_utils import grid_a_star, heuristic, create_grid, prune_path
from planning_utils import local_position_2_grid_coord, grid_coord_2_local_position
from planning_utils import graph_a_star, create_graph, closestNode
from planning_utils import get_receding_horizon_target, createProbabilisticRoadMap

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local, local_to_global

import matplotlib.pyplot as plt

class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

        self.home_lon = 0.0
        self.home_lat = 0.0

        self.grid = None
        self.north_offset = -1
        self.east_offset = -1

        self.SAFETY_DISTANCE = 5
        self.landing_alt = 0

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            localPos = np.array([self.local_position[0], self.local_position[1], -self.local_position[2]])
            dist2Target = np.linalg.norm(self.target_position - localPos)
            if len(self.waypoints) > 0:
                if dist2Target < self.SAFETY_DISTANCE:   # deadband safety distance
                    self.waypoint_transition()
                #else:
                #    self.navigate_to_target(dist2Target)
            else:
                if dist2Target < 1.0:   # deadband of 1 meter at the goal
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()
                #else:
                #    self.navigate_to_target()

    def velocity_callback(self):     # Can now land on top of a building
        if self.flight_state == States.LANDING:
            if np.abs(self.local_position[2] + self.landing_alt) < 0.1 or \
               np.linalg.norm(self.local_velocity) < 1.0:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()


    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], 
                          0)

    def navigate_to_target(self, dist2Target):
        receding_horizon_target = get_receding_horizon_target(self.grid, 
                                                              self.north_offset, self.east_offset,
                                                              self.local_position, self.target_position,
                                                              dist2Target, self.SAFETY_DISTANCE)
        print('local_position', self.local_position[0], self.local_position[1], -self.local_position[2])
        print('target_position', self.target_position[0], self.target_position[1], self.target_position[2])
        print('cmd_position', receding_horizon_target[0], receding_horizon_target[1], receding_horizon_target[2])
        self.cmd_position(receding_horizon_target[0], receding_horizon_target[1], 
                          self.target_position[2], 0)

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        #print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], 
                          0)

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING

        # TODO: Set global home position for the drone
        self.set_home_position(self.home_lon, self.home_lat, 0)

        # TODO: retrieve current global position - NOT NECESSARY
        # globalPos = self.global_position
 
        # TODO: convert to current local position using global_to_local() - NOT NECESSARY
        # localPos = global_to_local(global_position, self.global_home)

        print('global home {0}, position {1}, local position {2}'.format(self.global_home, 
                                                                            self.global_position,
                                                                            self.local_position))

        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection to begin navigation")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()

def select_grid_start_and_goal(grid, north_offset, east_offset, SAFETY_DISTANCE):
    grid_start = local_position_2_grid_coord((0, 0, 0), grid, north_offset, east_offset)

    plt.imshow(grid, origin='lower', cmap='Greys') 
    plt.plot(grid_start[1], grid_start[0], 'rx')

    plt.xlabel('EAST')
    plt.ylabel('NORTH')
    plt.title('Select goal position. Start position is shown.')
    pt_goal = plt.ginput(1, timeout=0)[0]
    plt.show(block=False)
    plt.close()

    # TODO: convert pts to grid values
    grid_goal = (int(pt_goal[1]), int(pt_goal[0]))

    grid_start_3d = (grid_start[0], grid_start[1], SAFETY_DISTANCE*2)

    landing_alt = np.int(grid[grid_goal[0], grid_goal[1]])
    grid_goal_3d = (grid_goal[0], grid_goal[1], landing_alt+SAFETY_DISTANCE*2)

    return grid_start_3d, grid_goal_3d, landing_alt

def plan_highlevel_path_using_grid_Astar(grid, north_offset, east_offset, 
                                         SAFETY_DISTANCE,
                                         grid_start, grid_goal):
    print("Searching for a path ...")        

    start = time.time()

    # Run A* to find a path from start to goal
    # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
    # or move to a different search space such as a graph (not done here)
    path, _ = grid_a_star(grid, heuristic, 
                          (grid_start[0], grid_start[1]), (grid_goal[0], grid_goal[1]), 
                          SAFETY_DISTANCE)

    if len(path) > 0:
        # altitude targets for smoothly transitioning to target altitude
        alts = np.linspace(grid_start[2], grid_goal[2], len(path))
        # Reach final altitude in free space one way point before the goal 
        alts[-2] = grid_goal[2]

        # construct 3d path points
        path = [[int(p[0]), int(p[1]), int(alt)] for p, alt in zip(path, alts)]

        # TODO: prune path to minimize number of waypoints
        print("Pruning the path ...")
        path = prune_path(path, grid, SAFETY_DISTANCE)

        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, p[2]] for p in path]
    else:
        waypoints = []

    end = time.time()
    print('Time taken by grid Astar planning: ', end - start)

    plt.imshow(grid, origin='lower', cmap='Greys') 

    if len(path) > 0:
        prev_pt = grid_start
        for pt in path:
            plt.plot([prev_pt[1], pt[1]], [prev_pt[0], pt[0]], 'r-')
            prev_pt = pt
    
        plt.plot(grid_start[1], grid_start[0], 'rx')
        plt.plot(grid_goal[1], grid_goal[0], 'rx')

        plt.xlabel('EAST')
        plt.ylabel('NORTH')
        plt.title('Close this figure when ready to begin navigation')
        plt.show()

    return waypoints


def plan_highlevel_path_using_graph_Astar(data, grid, north_offset, east_offset, 
                                          SAFETY_DISTANCE,
                                          grid_start, grid_goal):
    print("Searching for a path ...")
        
    # Define a graph for a particular altitude and safety margin around obstacles
    graph = create_graph(data, grid, north_offset, east_offset, SAFETY_DISTANCE)

    # define graph start and graph goal
    graph_start = closestNode(graph, (grid_start[0], grid_start[1]))
    graph_goal = closestNode(graph, (grid_goal[0], grid_goal[1]))

    start = time.time()

    # Run A* to find a path from start to goal
    # TODO: move to a different search space such as a graph (not done here)
    foundPath2Goal, path, _ = graph_a_star(graph, heuristic, graph_start, graph_goal)

    if len(path) > 0:
        # altitude targets for smoothly transitioning to target altitude
        alts = np.linspace(grid_start[2], grid_goal[2], len(path))

        # construct 3d path points
        path = [[p[0], p[1], int(alt)] for p, alt in zip(path, alts)]

        # insert the grid_start into the graph-based path
        path.insert(0, list(grid_start))
    
        if not foundPath2Goal:
            # We have a disconnected graph, 
            # let's join the disconnected sections using probabilistic roadmap planning
            foundPath2Goal, patchPath = createProbabilisticRoadMap(grid, tuple(path[-1]), grid_goal, SAFETY_DISTANCE)
            if not foundPath2Goal:
                print('**********************')
                print('Could not find even a probabilistic roadmap. This goal is unreachable!')
                print('**********************') 
                return []

            patchPath = [[int(p[0]), int(p[1]), int(p[2])] for p in patchPath]
            path += patchPath

        # put the grid goal into the graph-based path
        path.append(list(grid_goal))

        # TODO: prune path to minimize number of waypoints
        print("Pruning the path ...")
        path = prune_path(path, grid, SAFETY_DISTANCE)

        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, p[2]] for p in path]
    else:
        waypoints = []

    end = time.time()
    print('Time taken by graph Astar planning: ', end - start)

    if len(path) > 0:
        print('Creating map showing planned route...')

        plt.imshow(grid, origin='lower', cmap='Greys') 

        for e in graph.edges:
            p1 = e[0]
            p2 = e[1]
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'b-')
    
        prev_pt = grid_start
        for pt in path:
            plt.plot([prev_pt[1], pt[1]], [prev_pt[0], pt[0]], 'r-')
            prev_pt = pt
    
        plt.plot(grid_start[1], grid_start[0], 'rx')
        plt.plot(grid_goal[1], grid_goal[0], 'rx')

        plt.xlabel('EAST')
        plt.ylabel('NORTH')
        plt.title('Close this figure when ready to begin navigation')
        plt.show()
    
    return waypoints


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()
    
    SAFETY_DISTANCE = 3

    # TODO: read lat0, lon0 from colliders into floating point values
    firstRowData = np.genfromtxt('colliders.csv', delimiter=',', dtype=None, max_rows=1, encoding=None)
    lat0 = float(firstRowData[0].split()[1])
    lon0 = float(firstRowData[1].split()[1])
    global_home = np.array([lon0, lat0, 0])

    # Read in obstacle map
    data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

    # Define a grid for a particular altitude and safety margin around obstacles
    grid, north_offset, east_offset = create_grid(data, SAFETY_DISTANCE)
    print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))

    # select grid start and goal. 
    # goal is a randomly picked lat/lon converted to grid coordinates
    grid_start_3d, grid_goal_3d, landing_alt = \
        select_grid_start_and_goal(grid, north_offset, east_offset, SAFETY_DISTANCE)
    print('Grid Start and Goal: ', grid_start_3d, grid_goal_3d)

    waypoints = []

    print('Planning a path using graph Astar')
    waypoints = plan_highlevel_path_using_graph_Astar(data, grid, north_offset, 
                                                        east_offset, 
                                                        SAFETY_DISTANCE, grid_start_3d, 
                                                        grid_goal_3d)

    #if len(waypoints) == 0:
    #    print('Planning a path using grid Astar')
    #    waypoints = plan_highlevel_path_using_grid_Astar(grid, north_offset, 
    #                                                        east_offset, 
    #                                                        SAFETY_DISTANCE, grid_start_3d, 
    #                                                        grid_goal_3d)

    if len(waypoints) > 0:
        conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
        drone = MotionPlanning(conn)
        time.sleep(1)

        # initialize drone safety distance (for use in receding horizon calcs)
        drone.SAFETY_DISTANCE = SAFETY_DISTANCE

        # store home position (drone home position is set later in plan_path from these values)
        drone.home_lon = lon0
        drone.home_lat = lat0

        # store grid info for receding horizon calculations while navigating waypoints
        drone.grid = grid
        drone.north_offset = north_offset
        drone.east_offset = east_offset
         
        # Set drone waypoints
        drone.waypoints = waypoints

        # set initial target altitude for takeoff and landing alt (since we could land on top of a building)
        drone.target_position[2] = SAFETY_DISTANCE
        drone.landing_alt = landing_alt

        drone.start()
