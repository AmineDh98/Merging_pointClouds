import os
import numpy as np
from icp import *
from PIL import Image
import pandas as pd
from pyproj import Proj, transform
from plyfile import PlyData, PlyElement

class pointCloudMerging:
    def __init__(self):
        
        #######DEV0################
        # Assuming you have radial distortion coefficients k1 and k2
        self.k1_d0 = 0.022535963351356
        self.k2_d0 = -0.021289955281333

        # Calibration parameters for DEV0 (same as provided before)
        self.R_cam_lidar_DEV0 = np.array([[0.951099839323371,0.023671034661576,0.307975287575331],
                                    [-0.308201037834387,0.006476788999821,0.951299201871871],
                                    [0.020523545426254,-0.999698821306849,0.013455510426372]])

        self.t_cam_lidar_DEV0 = np.array([-0.002222602868500,-0.079224643637628,-0.115061940654467])

        self.K_cam_DEV0 = np.array([[1.296017692307357e+03, 0, 9.407268031732034e+02],
                                [0, 1.294832210476451e+03, 5.837191315595016e+02],
                                [0, 0, 1]])
        

        #######DEV1##################
        self.k1_d1 = 0.013878950622990
        self.k2_d1 = -0.010243402725356
        # Calibration parameters for DEV1 (same as provided before)
        self.R_cam_lidar_DEV1 = np.array([[0.926556230021193,0.003578990734975,-0.376139260692411],
                                          [0.376112846636539,-0.024011474677662,0.926262692587132],
                                          [-0.005716572738468,-0.999705276523598,-0.023594085848002]])

        self.t_cam_lidar_DEV1 = np.array([-0.020860151530965,-0.034278317810143,-0.142386558456705])

        self.K_cam_DEV1 =np.array([[1.282635220934342e+03,0,9.604166763029937e+02],
                        [0,1.282748868123230e+03,6.369097917615544e+02],
                        [0,0,1]])
        
        #######DEV2##################
        self.k1_d2 = 0.001222597654913
        self.k2_d2 = 3.749511060140799e-04
        # Calibration parameters for DEV2 (same as provided before)
        self.R_cam_lidar_DEV2 = np.array([[0.480481616111347,0.018925068602108,-0.876800580723709],
                                          [0.876808617497410,0.010779362579882,0.480718684497132],
                                          [0.018548985452295,-0.999762795427384,-0.011414377684026]])

        self.t_cam_lidar_DEV2 = np.array([0.010635763144944,-0.050409799554445,-0.165858222337824])

        self.K_cam_DEV2 =np.array([[1.273330078217040e+03,0,9.532565260903657e+02],
                                    [0,1.273507450812811e+03,6.002964464226565e+02],
                                      [0,0,1]])
        

        #######DEV3##################
        self.k1_d3 = 0.029246797758214
        self.k2_d3 = -0.009967784293123
        # Calibration parameters for DEV3 (same as provided before)
        self.R_cam_lidar_DEV3 = np.array([[0.544594066688672,-0.023325729033565,0.838375341295620],
                                          [-0.838651806756056,-0.004455026027122,0.544649703725156],
                                          [-0.008969367441335,-0.999717991789760,-0.021988345539215]])

        self.t_cam_lidar_DEV3 = np.array([-0.040088038110821,-0.031807930612324,-0.077223703086352])

        self.K_cam_DEV3 =np.array([[1.326676096357916e+03,0,9.455291558798021e+02],
                                    [0,1.326839969999406e+03,6.635059796083482e+02],
                                      [0,0,1]])


    # Function to read a point cloud from a file
    def read_point_cloud(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        points = []
        for line in lines:
            if not line.startswith('#'):  # Ignore comments
                values = [float(val) for val in line.split()[:3]]
                points.append(values)
        return np.array(points)
    
    def load_GPS(self, path_name):
        # Load GPS data using Pandas
        gps_data = pd.read_csv(path_name, delimiter=';')

        # Filter rows based on the 'ID' column
        lidar_id_range = range(1700, 1720)
        selected_data = gps_data[gps_data['ID'].isin(lidar_id_range)][['Lat', 'Lon', 'Alt']].values

        # If you want the result to be of size (20, 3)
        result_array = np.zeros((20, 3))
        result_array[:selected_data.shape[0]] = selected_data

        # Define the WGS84 and UTM projections
        wgs84 = Proj(init='epsg:4326')  # WGS84 coordinate system
        utm = Proj(init='epsg:32634')  # Budapest is in UTM Zone 34T
        transformed_coordinates = []
        for lat, lon, alt in result_array:
            x, y, z = transform(wgs84, utm, lon, lat, alt, radians=False)
            transformed_coordinates.append([x, y, z])
        
        # Convert to NumPy array for plotting
        transformed_coordinates = np.array(transformed_coordinates)
        return transformed_coordinates
    
    def gps_to_cartesian(self, latitude, longitude, altitude):
        """
        Convert GPS coordinates to Cartesian coordinates (x, y, z).

        Parameters:
        - latitude (float): Latitude in decimal degrees.
        - longitude (float): Longitude in decimal degrees.
        - altitude (float): Altitude in meters.

        Returns:
        - np.array: Cartesian coordinates (x, y, z).
        """
        # Earth radius in meters (assuming a spherical Earth model)
        earth_radius = 6371000.0

        # Convert latitude and longitude to radians
        lat_rad = np.radians(latitude)
        lon_rad = np.radians(longitude)

        # Calculate Cartesian coordinates
        x = earth_radius * np.cos(lat_rad) * np.cos(lon_rad)
        y = earth_radius * np.cos(lat_rad) * np.sin(lon_rad)
        z = earth_radius * np.sin(lat_rad) + altitude

        return np.array([x, y, z])
    
    def write_ply(self,file_path, local, param):
        """
        Write a NumPy array of points to a PLY file.

        Parameters:
        - file_path (str): The path to the PLY file.
        - points (np.array): A NumPy array of shape (num_of_pointClouds, 3).

        Returns:
        - None
        """
        
            
        num_points = local.shape[0]

        with open(file_path, 'w') as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex {}\n".format(num_points))
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            # Write point cloud data
            for i in range(num_points):
                if param==3:
                    f.write(
                        str(local[i, 0])
                        + " "
                        + str(local[i, 1])
                        + " "
                        + str(local[i, 2])
                        + " "
                        + '0'
                        + " "
                        + '0'
                        + " "
                        + '0'
                        + "\n"
                    )
                else:
                    # f.write("{} {} {} {} {} {}\n".format(local[i, 0], local[i, 1], local[i, 2], local[i, 3], local[i, 4], local[i, 5]))
                    f.write(
                        str(local[i, 0])
                        + " "
                        + str(local[i, 1])
                        + " "
                        + str(local[i, 2])
                        + " "
                        + str(int(local[i, 3]))
                        + " "
                        + str(int(local[i, 4]))
                        + " "
                        + str(int(local[i, 5]))
                        + "\n"
                    )



    
    def read_camera_images(self, folder_path, n, num_images=20):
        """
        Reads camera images from a specified folder.

        Parameters:
            - folder_path: Path to the folder containing camera images.
            - num_images: Number of images to read (default is 20).

        Returns:
            - images: List of images as NumPy arrays.
        """
        
        images = []
        
        for i in range(num_images):
            if i <10:
                image_path = os.path.join(folder_path, f'Dev{n}_Image_w1920_h1200_fn170{i}.jpg')
            else:
                image_path = os.path.join(folder_path, f'Dev{n}_Image_w1920_h1200_fn17{i}.jpg')

            image = cv2.imread(image_path)

            # Optionally, you can convert the image to RGB if needed
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if image is not None:
                images.append(image)
            else:
                print(f"Error reading image: {image_path}")
            

        return images

    # Function to merge point clouds using ICP
    def merge_point_clouds(self, global_point_clouds, number):
        # Read the first point cloud as the initial source
        source = global_point_clouds[0,:,:]
        final = [source]

        # Iterate through the rest of the point clouds
        for i in range(1, number):
            target = global_point_clouds[i,:,:]

            # Apply ICP to align the source and target point clouds
            source = icp(source, target)
            final.append(source)

        return np.array(final)
    
    def GPS_transformation(self,file_paths,  transformed_coordinates):
        global_point_clouds = []
        for i in range(0, len(file_paths)):
            local_map = self.read_point_cloud(file_paths[i])
            # Compute mean position of the local point cloud
            mean_position = np.mean(local_map, axis=0)
            
            # Compute translation vector from GPS position to mean position
            translation_vector = mean_position - transformed_coordinates[i]
            
            # Translate all points in the local_map to the world frame
            local_map_transformed = local_map - translation_vector
            # Update the local_map with the transformed coordinates
            global_point_clouds.append(local_map_transformed)
        return np.array(global_point_clouds)
    
    def project_to_image(self, uncolored_points, dev):
        projected_point_clouds = []
        # Transform each LiDAR point and project it onto the image plane for DEV0
        if dev ==0:
            # Create a 4x4 identity matrix
            transformation_matrix = np.eye(4)
            # Fill the top-left 3x3 submatrix with the rotation matrix
            transformation_matrix[:3, :3] = self.R_cam_lidar_DEV0
            # Fill the last column with the translation vector
            transformation_matrix[:3, 3] = self.t_cam_lidar_DEV0
            k1 = self.k1_d0
            k2 = self.k2_d0
            K=self.K_cam_DEV0
        elif dev==1:
            # Create a 4x4 identity matrix
            transformation_matrix = np.eye(4)
            # Fill the top-left 3x3 submatrix with the rotation matrix
            transformation_matrix[:3, :3] = self.R_cam_lidar_DEV1
            # Fill the last column with the translation vector
            transformation_matrix[:3, 3] = self.t_cam_lidar_DEV1
            k1 = self.k1_d1
            k2 = self.k2_d1
            K=self.K_cam_DEV1
        elif dev==2:
            # Create a 4x4 identity matrix
            transformation_matrix = np.eye(4)
            # Fill the top-left 3x3 submatrix with the rotation matrix
            transformation_matrix[:3, :3] = self.R_cam_lidar_DEV2
            # Fill the last column with the translation vector
            transformation_matrix[:3, 3] = self.t_cam_lidar_DEV2
            k1 = self.k1_d2
            k2 = self.k2_d2
            K=self.K_cam_DEV2
        elif dev==3:
            # Create a 4x4 identity matrix
            transformation_matrix = np.eye(4)
            # Fill the top-left 3x3 submatrix with the rotation matrix
            transformation_matrix[:3, :3] = self.R_cam_lidar_DEV3
            # Fill the last column with the translation vector
            transformation_matrix[:3, 3] = self.t_cam_lidar_DEV3
            k1 = self.k1_d3
            k2 = self.k2_d3
            K=self.K_cam_DEV3
                
            

        for point_cloud in uncolored_points:
            
            homogeneous_point_cloud = np.append(point_cloud, 1) # homogeneous coordinates

            
            # Transform LiDAR point to camera coordinates for DEV0
            point_cloud_cam_DEV0 = np.linalg.inv(transformation_matrix) @ homogeneous_point_cloud

            # Apply radial distortion correction
            r = np.linalg.norm(point_cloud_cam_DEV0[:2])
            factor = 1 + k1 * r**2 + k2 * r**4
            
            # Apply correction to the coordinates
            point_cloud_cam_DEV0[:2] *= factor

            # Project LiDAR point onto the image plane for DEV0
            pixel_coordinates_DEV0 = K @ point_cloud_cam_DEV0[:3]
            
            pixel_coordinates_DEV0 /= pixel_coordinates_DEV0[2]  # Normalize by z coordinate for pixel coordinates (u, v)
            projected_point_clouds.append(pixel_coordinates_DEV0[:2])
        return np.array(projected_point_clouds)

        
    def coloring(self, projected_point_clouds, image, uncolored_points):
        done=[]
        colored_point_clouds = []
        for ind,( im,projected) in enumerate(zip(image,projected_point_clouds)):
            if ind ==0:
                for index, (point2d, point3d) in enumerate(zip(projected,uncolored_points)):
                    if ((round(point2d[0]) in range (0,1200)) and (round(point2d[1]) in range (0,1920))):
                        rgb =  im[0][round(point2d[0]),round(point2d[1]),:]
                        colored_point = np.array([point3d[0], point3d[1], point3d[2],rgb[0], rgb[1], rgb[2]])
                        done.append(index)
                        colored_point_clouds.append(colored_point)
                    else:
                        rgb = np.array([0,0,0]) 
                        colored_point = np.array([point3d[0], point3d[1], point3d[2],rgb[0], rgb[1], rgb[2]])

                        colored_point_clouds.append(colored_point)
            else:
                for index, (point2d, point3d) in enumerate(zip(projected,uncolored_points)):
                    if index not in done:
                        if ((round(point2d[0]) in range (0,1200)) and (round(point2d[1]) in range (0,1920))):
                            rgb =  im[0][round(point2d[0]),round(point2d[1]),:]
                            colored_point = np.array([point3d[0], point3d[1], point3d[2],rgb[0], rgb[1], rgb[2]])
                            done.append(index)
                            colored_point_clouds[index] = colored_point
        return np.array(colored_point_clouds)
    


    def read_ply(self, file_path):
        """
        Read a PLY file and return the data as a NumPy array.

        Parameters:
        - file_path (str): The path to the PLY file.

        Returns:
        - np.array: A NumPy array of shape (num_of_pointClouds, 3) containing the point cloud data.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Find the index where the data starts in the PLY file
        data_start_index = lines.index("end_header\n") + 1

        # Read point cloud data from the lines
        point_cloud_data = [list(map(float, line.split())) for line in lines[data_start_index:]]
        
        return np.array(point_cloud_data)
    
    def normalize_gps_coordinates(self, coordinates):
        """
        Normalize GPS coordinates.

        Parameters:
        - coordinates (np.array): GPS coordinates array of shape (n, 3).

        Returns:
        - np.array: Normalized GPS coordinates array.
        """
        # Calculate mean and standard deviation for each coordinate
        mean_values = np.mean(coordinates, axis=0)
        std_dev_values = np.std(coordinates, axis=0)

        # Subtract the mean and divide by standard deviation
        normalized_coordinates = (coordinates - mean_values) / std_dev_values

        return normalized_coordinates
    
    
    
                


    def main(self):
        # # # Directory containing the point cloud files
        data_dir = r'C:\Users\Emin\Desktop\3D_sensors_and_sensor_fusion\ICP\20230918\data'

        transformed_coordinates = self.load_GPS('20230918_ELTEkorV3.6_RTK.csv')
        normalized_coordinates = self.normalize_gps_coordinates(transformed_coordinates)
        print('transformed_coordinates',transformed_coordinates)
        # List all files in the directory with the .xyz extension
        file_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".xyz")]
        number = len(file_paths)
        global_point_clouds = self.GPS_transformation(file_paths,  normalized_coordinates)
        print('global_point_clouds after GPS transformation')
        print(global_point_clouds.shape)
        # print(global_point_clouds[0,:20,:])
        # Merge point clouds using ICP
        merged_point_cloud = self.merge_point_clouds(global_point_clouds, number)
        merged_point_cloud = merged_point_cloud.reshape(-1, 3)
        # Write the merged point cloud to a new file

        self.write_ply('merged_point_cloud.ply', merged_point_cloud, 3)
        # uncolored_points = self.read_ply('merged_point_cloud_middle.ply')
        uncolored_points = merged_point_cloud
        print('file has been red')
        print(uncolored_points.shape)
        all_projected=[]
        for dev in range(4):
            projected_point_clouds = self.project_to_image(uncolored_points, dev)
            all_projected.append(projected_point_clouds)
        # print('projected_point_clouds')
        # print(projected_point_clouds.shape)
        

        images0 = self.read_camera_images(r'C:\Users\Emin\Desktop\3D_sensors_and_sensor_fusion\ICP\20230918\images\DEV0',0)
        images1 = self.read_camera_images(r'C:\Users\Emin\Desktop\3D_sensors_and_sensor_fusion\ICP\20230918\images\DEV1',1)
        images2 = self.read_camera_images(r'C:\Users\Emin\Desktop\3D_sensors_and_sensor_fusion\ICP\20230918\images\DEV2',2)
        images3 = self.read_camera_images(r'C:\Users\Emin\Desktop\3D_sensors_and_sensor_fusion\ICP\20230918\images\DEV3',3)
        images = [images0,images1,images2,images3]

        colored_point_clouds = self.coloring(all_projected,images, uncolored_points)
        print('colored_point_clouds')
        print(colored_point_clouds.shape)
        self.write_ply('colored_merged_point_cloud.ply', colored_point_clouds, 6)


if __name__ == "__main__":
    merge = pointCloudMerging()
    merge.main()
    
