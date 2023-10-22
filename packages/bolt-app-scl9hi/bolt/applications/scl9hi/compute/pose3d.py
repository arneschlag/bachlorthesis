from bolt.data.containers.attributes import FloatAttribute
from bolt.applications.pose_estimation_3d.data.keypoint import PoseKeypoints
from bolt.applications.keypoint_detection.data.keypoint import KeypointsDefinition
import numpy as np

class Pose3D(FloatAttribute, PoseKeypoints):
    def __init__(
        self,
        kp_df: KeypointsDefinition,
        name: str=None,
        is_abs: bool = False
    ):
        super().__init__(
            name=name,
            length=kp_df.num_keypoints * 3,
            ignore_value=np.nan,
        )
        # init empty array
        ndarray = np.empty((kp_df.num_keypoints,3))
        ndarray[:] = np.nan
        self.kp_df = kp_df
        PoseKeypoints.__init__(self, ndarray, kp_df, kp_df.root_kp_name)

        self.num_of_keypoints = int(self.length / 3)
        self.symmetric_keypoint_pairs = kp_df.flip_keypoints_map
    def print_attribute_info(self):
        """
        Returns some general information about the attribute.

        This is also the string that will be printed by PickleAnnotationLoader at the end
        of data generation (e.g. "3 classes").

        Returns
        -------
        str
        """
        return f"{self.name} ({self.num_of_keypoints} keypoints)"

    def annotation_2_ndarray(self, annotation, key):
        """
        Converts the annotation's attribute value to an ndarray.

        The format of attribute annotation is an array in the form ((x1,y1,z1),(x2,y2,z2),....)
        where xi, yi and zi are the coordinates of the keypoint

        returns in the following format (x1,y1,z1,x2,y2,z2, ...)
        Parameters
        ----------
        annotation : dict
            Dictionary of annotations
        key : str
            A key in annotation where the attribute is located

        Returns
        -------
        ndarray
        """
        if annotation.get(key, None) is None:
            attribute = [self.ignore_value] * 3 * self.num_of_keypoints
        else:
            attribute = annotation[key]
        nd_array = np.array(attribute, dtype=np.float32).flatten()

        #keypoint_data : np.ndarray
        #    Numpy array of shape (B, J, D), where D is usually 2 or 3.
        #    B can be 1.

        # detect if the structure is x,y,z or x,y,z, a, where a
        # denotates the occlusion, for the 3d kp we ignore this
        if ((len(nd_array) % 4) == 0 and (len(nd_array) % 3) != 0) or sum(nd_array[3::4] % 1) == 0:
            # it is x,y,z,a or it is probably  x,y,z,a
            np_list = list(nd_array)
            del np_list[3::4]
            nd_array = np.array(np_list)


        bjd = nd_array.reshape((1,int(len(nd_array)/3), 3))
        PoseKeypoints.__init__(self, bjd, self.kp_df, self.kp_df.root_kp_name)
        root_relative = self.root_relative_coordinates().flatten()
        return root_relative
    def get_abs_coord(self,annotation, key):
        if annotation.get(key, None) is None:
            attribute = [self.ignore_value] * 3
        else:
            attribute = annotation[key]
        nd_array = np.array(attribute, dtype=np.float32).flatten()
        if ((len(nd_array) % 4) == 0 and (len(nd_array) % 3) != 0) or sum(nd_array[3::4] % 1) == 0:
            # it is x,y,z,a or it is probably  x,y,z,a
            np_list = list(nd_array)
            del np_list[3::4]
            nd_array = np.array(np_list)


        bjd = nd_array.reshape((1,int(len(nd_array)/3), 3))
        PoseKeypoints.__init__(self, bjd, self.kp_df, self.kp_df.root_kp_name)
        abs_coord = self.arr[:, self.root_keypoint_idx, :]
        return np.array(abs_coord, dtype=np.float32).flatten()
    def empty_ndarray_abs(self,size):
        return np.ones((size, 3), dtype=np.float32)

class RefPoint(FloatAttribute):
    def __init__(
        self,
        kp_df: KeypointsDefinition,
        name: str=None,
    ):
        super().__init__(
            name=name,
            length=3,
            ignore_value=np.nan,
        )
        # init empty array
        ndarray = np.empty((1,3))
        ndarray[:] = np.nan
        self.kp_df = kp_df

        self.num_of_keypoints = 1
    def print_attribute_info(self):
        """
        Returns some general information about the attribute.

        This is also the string that will be printed by PickleAnnotationLoader at the end
        of data generation (e.g. "3 classes").

        Returns
        -------
        str
        """
        return f"{self.name} ({self.num_of_keypoints} keypoints)"

    def annotation_2_ndarray(self, annotation, key):
        """
        Converts the annotation's attribute value to an ndarray.

        The format of attribute annotation is an array in the form ((x1,y1,z1),(x2,y2,z2),....)
        where xi, yi and zi are the coordinates of the keypoint

        returns in the following format (x1,y1,z1,x2,y2,z2, ...)
        Parameters
        ----------
        annotation : dict
            Dictionary of annotations
        key : str
            A key in annotation where the attribute is located

        Returns
        -------
        ndarray
        """
        if annotation.get(key, None) is None:
            attribute = [self.ignore_value] * 3 * self.num_of_keypoints
        else:
            attribute = annotation[key]
        nd_array = np.array(attribute, dtype=np.float32).flatten()

        #keypoint_data : np.ndarray
        #    Numpy array of shape (B, J, D), where D is usually 2 or 3.
        #    B can be 1.

        # detect if the structure is x,y,z or x,y,z, a, where a
        # denotates the occlusion, for the 3d kp we ignore this
        if ((len(nd_array) % 4) == 0 and (len(nd_array) % 3) != 0) or sum(nd_array[3::4] % 1) == 0:
            # it is x,y,z,a or it is probably  x,y,z,a
            np_list = list(nd_array)
            del np_list[3::4]
            nd_array = np.array(np_list)


        bjd = nd_array.reshape((1,int(len(nd_array)/3), 3))
        PoseKeypoints.__init__(self, bjd, self.kp_df, self.kp_df.root_kp_name)
        root_relative = self.root_relative_coordinates().flatten()
        return root_relative
