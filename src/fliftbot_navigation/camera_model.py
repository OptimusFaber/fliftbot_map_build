from __future__ import annotations

"""
Camera Model for Ceiling Marker Navigation (OpenCV-based)

This module provides a camera model tailored for a robot that reads
navigation markers on the ceiling using an almost-vertical camera.

Architecture Overview:
	1. Initialization from an OpenCV-style camera json and physical mount
	   offsets/tilts relative to the robot coordinate frame
	2. Efficient projection of 3D world points to image pixels (vectorized)
	3. Back-projection of image pixels to world-frame rays (origin + unit dir)
	4. Analytical pixel Jacobians w.r.t. robot 6-DoF pose for uncertainty propagation

Key Features:
	- Simple, typed public API with numpy arrays
	- Uses OpenCV functions (Rodrigues, projectPoints, undistortPoints)
	- Fast array operations, minimal Python loops
	- Explicit handling of common input shapes for robot pose
	- Jacobian chaining via closed-form derivatives and OpenCV-provided projection Jacobian

Coordinate Frames:
	- World: Z up, X/Y horizontal; right-handed.
	- Robot: co-located axes with world but rotated by robot orientation; yaw is about +Z.
	- Camera: OpenCV convention; base orientation such that at zero tilt the optical axis
	  aligns with +World Z and image rows increase downward.

Conventions and Notation:
	- Robot pose q = [x, y, z, rx, ry, rz] where rx,ry,rz are Tait-Bryan angles (intrinsic X-Y-Z).
	- Projection uses world->camera transform: Xc = R_wc Xw + t, with t = -R_wc C_world.
	- Pixels u,v are obtained via cv2.projectPoints.
	- Jacobian J(u,v;q) is returned as shape (N, 2, 6) for N points; non-visible points have NaNs.
	- Chain rule: ∂I/∂q = (∂I/∂rvec)(∂rvec/∂R_wc)(∂R_wc/∂q) + (∂I/∂t)(∂t/∂q), where
	  ∂t/∂q = - (∂R_wc/∂q) C_world - R_wc (∂C_world/∂q) and ∂C_world/∂pos = I, ∂C_world/∂angles = (∂R_wr/∂angles) offsets.

Camera JSON:
	- width, height: image dimensions in pixels
	- focal_x, focal_y: normalized focal parameters (scaled by max(width,height) to pixels)
	- c_x, c_y: principal point offsets (normalized, scaled similarly)
	- k1, k2, p1, p2, k3: Brown distortion parameters (used for distorted projection)
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import cv2
# from line_profiler import profile

def intersect_plane(origin_pt: np.ndarray, dir_vec: np.ndarray, z_plane: float) -> np.ndarray:
	"""Intersect ray with z=constant plane and return the 3D point."""
	t = (z_plane - origin_pt[2]) / dir_vec[2]
	return origin_pt + dir_vec * t


def cone_gate_mask(camera_model: CameraModel,
				   robot_pos: np.ndarray,
				   robot_orient: np.ndarray,
				   points_3d: np.ndarray,
				   half_angle_cos: float = 0.5) -> np.ndarray:
	"""Return boolean mask selecting markers inside the upward-facing camera cone.

	Uses camera origin and optical axis in world coords derived from rvec/tvec.
	half_angle_cos=0.5 corresponds to 60° half-angle.
	"""
	rvec, tvec = camera_model.get_rvec_tvec(robot_pos=robot_pos, robot_orient=robot_orient)
	R_world2camera, _ = cv2.Rodrigues(rvec)
	R_camera2world = R_world2camera.T
	C_world = -(R_camera2world @ tvec.reshape(3))
	optical_axis_world = R_camera2world @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
	marker_vecs = np.asarray(points_3d, dtype=np.float64) - C_world[None, :]
	norms = np.linalg.norm(marker_vecs, axis=1)
	with np.errstate(invalid='ignore', divide='ignore'):
		cos_angles = (marker_vecs @ optical_axis_world) / norms
	gate_mask = np.isfinite(cos_angles) & (cos_angles >= half_angle_cos)
	return gate_mask


class CameraModel:
	"""Ceiling navigation camera model.

	Contract:
	- Inputs
		- offsets: (x, y, z) camera position relative to robot frame, meters
		- tilts: (rx, ry, rz) radians about robot X,Y,Z axes; (0,0,0) is vertical
		- camera_json_path: path to json containing camera intrinsics and distortion
		- camera_key: optional key to select a specific camera entry
	- Outputs
		- project_points: (N, 2) pixel coordinates per world point and (N,2,6) Jacobian w.r.t 6-DoF
		- pixels_to_rays: origin (3,), and (N, 3) unit direction vectors in world

	Coordinate frames:
		- World: Z up, X/Y horizontal; robot yaw rotates around +Z
		- Robot: same axes as world, positioned at robot base center
		- Camera: OpenCV convention; at zero tilt, optical axis aligns with +World Z and
		  image rows increase downward.
	"""

	def __init__(
		self,
		offsets: Tuple[float, float, float],
		tilts_rad: Tuple[float, float, float],
		camera_json_path: str,
		camera_key: Optional[str] = None,
	) -> None:
		self.offsets = np.asarray(offsets, dtype=float).reshape(3)
		self.tilts = np.asarray(tilts_rad, dtype=float).reshape(3)

		# Load camera parameters
		with open(camera_json_path, "r", encoding="utf-8") as f:
			cam_data: Dict[str, Dict[str, Any]] = json.load(f)

		if camera_key is None:
			# Use the first key if not specified
			if len(cam_data) == 0:
				raise ValueError("Camera JSON is empty.")
			camera_key = next(iter(cam_data.keys()))
		if camera_key not in cam_data:
			raise KeyError(f"Camera entry '{camera_key}' not found in JSON.")

		entry = cam_data[camera_key]
		self.width = int(entry.get("width"))
		self.height = int(entry.get("height"))

		# Build intrinsic matrix and distortion coefficients from JSON
		self.K_camera_matrix, self.D_distorsion_coefficients = self._intrinsics_from_json(entry)

		# Precompute mount/base rotations
		self.R_rot_matrix_mounting = self._rotation_matrix_from_euler(self.tilts)
		# Base orientation: identity — OpenCV conventions are handled via intrinsics
		self.R_base = np.eye(3, dtype=np.float64)

	# -------------------------------------------------------------------------------------
	# Public API
	# -------------------------------------------------------------------------------------


	# @profile
	def project_points(
		self,
		robot_pos: np.ndarray,
		robot_orient: np.ndarray,
		points_3d: np.ndarray,
		return_distorted: bool = True,
		return_visible: bool = True,
	) -> Tuple[np.ndarray, np.ndarray]:
		"""Project 3D world points to image pixels with visibility filtering and pose Jacobian.

		Args:
			robot_pos: Robot position in world; shape (2,) [x,y] or (3,) [x,y,z]. If (2,), z=0.
			robot_orient: Robot orientation; shape (1,) [yaw] or (3,) [rx,ry,rz] in radians.
			points_3d: World coordinates of points; shape (N,3).
			return_distorted: If True, use distortion coefficients for projection. If False, use ideal pinhole model.
			return_visible: If True, points outside camera FoV are set to (nan,nan). If False, all points projected.

			Warning: When return_visible is False and return_distorted is True, points ouside the FoV can produce invalid projections due to high order distorsion coefficients.

		Returns:
			pixels: (N,2) array of image pixel coordinates. Points outside the camera FoV
				(behind camera or outside image bounds) are returned as (nan,nan).
			jacobian: (N,2,6) per-point Jacobian ∂(u,v)/∂q with q=[x,y,z,rx,ry,rz].
				Computed only for visible points; non-visible rows are NaN.

		Jacobian details:
			Let Xc = R_wc Xw + t, pixels I = project(R_wc, t, K, D, Xw).
			OpenCV returns ∂I/∂rvec and ∂I/∂tvec from cv2.projectPoints. We chain via:
				∂I/∂q = (∂I/∂rvec)(∂rvec/∂R_wc)(∂R_wc/∂q) + (∂I/∂tvec)(∂t/∂q),
				with ∂t/∂q = - (∂R_wc/∂q) C_world - R_wc (∂C_world/∂q).
			For position:
				∂C_world/∂[x,y,z] = I_3, and rvec has no dependence on translation.
			For orientation (rx,ry,rz):
				R_wr = Rz @ Ry @ Rx; ∂R_wr/∂rx = Rz Ry ∂Rx, ∂R_wr/∂ry = Rz ∂Ry Rx, ∂R_wr/∂rz = ∂Rz Ry Rx.
				R_cw = R_wr M, R_wc = R_cw^T; thus ∂R_wc/∂⋅ = (∂R_cw/∂⋅)^T.
				∂C_world/∂rx = (∂R_wr/∂rx) offsets, similarly for ry,rz.
		Shapes:
			J_proj_rvec: (2N_v,3), J_proj_tvec: (2N_v,3), dr_dq: (3,6), dt_dq: (3,6),
			J_total (stacked over visible): (2N_v,6) → reshaped to (N_v,2,6) and scattered.
		"""
		P_robot, R_world2robot = self._robot_pose(robot_pos, robot_orient)
		C_world, R_camera2world = self._camera_world_pose(P_robot, R_world2robot)

		# OpenCV expects R and t that transform world points to camera coords:
		# Xc = R * Xw + t  with  t = -R * C_world
		R_world2camera = R_camera2world.T
		tvec = (-R_world2camera @ C_world.reshape(3, 1)).reshape(3)
		rvec, J_rvec_dR = cv2.Rodrigues(R_world2camera)
		# Rodrigues Jacobian layout note:
		# - OpenCV returns drvec/dR as (9,3) array.
		# - We standardize to (3,9) so we can compute (3,9) @ vec(dR) → (3,).
		if J_rvec_dR.shape == (9, 3):
			J_rvec_dR = J_rvec_dR.T

		obj_pts = np.asarray(points_3d, dtype=np.float64).reshape(-1, 1, 3)
		img_pts, jacobian_projection = cv2.projectPoints(obj_pts, rvec, tvec, self.K_camera_matrix, np.zeros(5))  # no distortion
		pixels = img_pts.reshape(-1, 2)

		if return_visible:
			# Visibility filtering: compute camera-frame depth and image bounds
			cam_coords = (R_world2camera @ np.asarray(points_3d, dtype=np.float64).T + tvec.reshape(3, 1)).T
			z = cam_coords[:, 2]
			w, h = int(self.width), int(self.height)
			u = pixels[:, 0]
			v = pixels[:, 1]
			in_frame = (z > 0.0) & np.isfinite(u) & np.isfinite(v) & (u >= 0.0) & (u < w) & (v >= 0.0) & (v < h)
			# Set out-of-FoV pixels to NaN
			pixels[~in_frame] = np.array([np.nan, np.nan], dtype=np.float64)
		else:
			in_frame = np.full((pixels.shape[0],), True, dtype=bool)
		# project the valid points with distortion
		pts_valid = obj_pts[in_frame].reshape(-1, 1, 3)
		# print(f"Projecting {pts_valid.shape[0]} valid points with distortion for Jacobian computation.")
		if return_distorted and np.any(in_frame):
			img_pts_distorted, jacobian_projection = cv2.projectPoints(
				pts_valid, rvec, tvec, self.K_camera_matrix, self.D_distorsion_coefficients
			)
			# OpenCV Jacobian columns (we use only the first 6):
			# [0:3] → ∂I/∂rvec, [3:6] → ∂I/∂tvec, remaining columns correspond to intrinsics/distortion.
			pixels[in_frame] = img_pts_distorted.reshape(-1, 2)

		# ---------------------------------------------------------------------------------
		# Jacobian w.r.t robot 6-DoF [x, y, z, rx, ry, rz]
		# Chain rule breakdown (per visible point):
		#   ∂I/∂q = (∂I/∂rvec)(∂rvec/∂R_wc)(∂R_wc/∂q) + (∂I/∂tvec)[ - (∂R_wc/∂q) C_world - R_wc (∂C_world/∂q) ]
		#   with ∂C_world/∂pos = I_3 and ∂C_world/∂angles = (∂R_wr/∂angles) offsets.
		# We compute ∂R_wr/∂rx, ∂ry, ∂rz analytically for R_wr = Rz @ Ry @ Rx and propagate.
		# ---------------------------------------------------------------------------------
		# Prepare output Jacobian (N, 2, 6), default NaNs
		J_out = np.full((pixels.shape[0], 2, 6), np.nan, dtype=np.float64)

		if np.any(in_frame):
			# Slice projectPoints Jacobian for visible points
			# jacobian_projection shape: (2*Nv, M). First 3 cols -> dI/drvec, next 3 -> dI/dtvec
			J_proj_r = jacobian_projection[:, 0:3]
			J_proj_t = jacobian_projection[:, 3:6]

			# Compute analytic pose derivatives
			# Normalize orientation to full (rx, ry, rz)
			orient = np.asarray(robot_orient, dtype=float)
			if orient.shape == (1,):
				angles = np.array([0.0, 0.0, float(orient[0])], dtype=np.float64)
			elif orient.shape == (3,):
				angles = orient.astype(np.float64)
			else:
				raise ValueError("robot_orient must be shape (1,) or (3,)")

			# Derivatives of R_wr = Rz @ Ry @ Rx (intrinsic X-Y-Z, Tait-Bryan)
			Rx, Ry, Rz = self._euler_component_matrices(angles)
			dRx, dRy, dRz = self._euler_component_derivatives(angles)
			R_wr = Rz @ Ry @ Rx
			M = self.R_rot_matrix_mounting @ self.R_base
			R_cw = R_wr @ M
			R_wc = R_cw.T

			# dR_cw/d{rx,ry,rz} via product rule
			dR_wr_rx = Rz @ Ry @ dRx
			dR_wr_ry = Rz @ dRy @ Rx
			dR_wr_rz = dRz @ Ry @ Rx
			dR_cw_rx = dR_wr_rx @ M
			dR_cw_ry = dR_wr_ry @ M
			dR_cw_rz = dR_wr_rz @ M
			# dR_wc/d* = (dR_cw)^T since R_wc = (R_cw)^T
			dR_wc_rx = dR_cw_rx.T
			dR_wc_ry = dR_cw_ry.T
			dR_wc_rz = dR_cw_rz.T

			# ∂C_world/∂{pos} and ∂C_world/∂{angles}
			dC_dpos = np.eye(3, dtype=np.float64)  # x,y,z
			off = self.offsets.reshape(3)
			dC_drx = dR_wr_rx @ off  # ∂C/∂rx = (∂R_wr/∂rx) offsets
			dC_dry = dR_wr_ry @ off  # ∂C/∂ry = (∂R_wr/∂ry) offsets
			dC_drz = dR_wr_rz @ off  # ∂C/∂rz = (∂R_wr/∂rz) offsets

			# ∂t/∂q derivatives
			dt_dpos = -R_wc @ dC_dpos  # ∂t/∂pos = -R_wc ∂C/∂pos = -R_wc I_3
			dt_drx = -(dR_wc_rx @ C_world + R_wc @ dC_drx)
			dt_dry = -(dR_wc_ry @ C_world + R_wc @ dC_dry)
			dt_drz = -(dR_wc_rz @ C_world + R_wc @ dC_drz)

			# ∂rvec/∂q via Rodrigues Jacobian (drvec/dR) and ∂R_wc/∂q
			# Flatten dR_wc_* (row-major) to match our (3,9) layout for J_rvec_dR
			dr_drx = (J_rvec_dR @ dR_wc_rx.reshape(9))  # (3,)
			dr_dry = (J_rvec_dR @ dR_wc_ry.reshape(9))
			dr_drz = (J_rvec_dR @ dR_wc_rz.reshape(9))

			# Assemble [x,y,z, rx,ry,rz] derivatives for rvec and tvec
			dr_dq = np.column_stack([
				np.zeros(3, dtype=np.float64),  # dx (rvec independent of translation)
				np.zeros(3, dtype=np.float64),  # dy
				np.zeros(3, dtype=np.float64),  # dz
				dr_drx,
				dr_dry,
				dr_drz,
			])  # (3x6)
			dt_dq = np.column_stack([
				dt_dpos[:, 0],
				dt_dpos[:, 1],
				dt_dpos[:, 2],
				dt_drx,
				dt_dry,
				dt_drz,
			])  # (3x6)

			# Chain: (2N_v x 3)@(3x6) → (2N_v x 6), then reshape per-point
			J_total = J_proj_r @ dr_dq + J_proj_t @ dt_dq  # (2Nv,6)
			J_total = J_total.reshape(-1, 2, 6)  # (Nv,2,6)

			# Scatter into full-sized output using mask
			J_out[in_frame] = J_total

		return pixels, J_out

	def get_rvec_tvec(
		self,
		robot_pos: np.ndarray,
		robot_orient: np.ndarray,
	) -> Tuple[np.ndarray, np.ndarray]:
		"""Compute rotation vector (rvec) and translation vector (tvec) for world->camera.

		These map world points Xw to camera points Xc via Xc = R * Xw + t.
		The rotation is returned in Rodrigues vector form (OpenCV convention).
		"""
		P_robot, R_world2robot = self._robot_pose(robot_pos, robot_orient)
		C_world, R_camera2world = self._camera_world_pose(P_robot, R_world2robot)
		R_world2camera = R_camera2world.T
		tvec = (-R_world2camera @ C_world.reshape(3, 1)).reshape(3)
		rvec, _ = cv2.Rodrigues(R_world2camera)
		return rvec.reshape(3), tvec.reshape(3)

	def pixels_to_rays(
		self,
		robot_pos: np.ndarray,
		robot_orient: np.ndarray,
		pixels: np.ndarray,
	) -> Tuple[np.ndarray, np.ndarray]:
		"""Convert image pixels to world-frame rays.

		Args:
			robot_pos: np.array of shape (2,) or (3,) — robot position in world.
			robot_orient: np.array of shape (1,) or (3,) — robot orientation.
			pixels: np.array of shape (N, 2) — image pixel coordinates.

		Returns:
			origin: np.array of shape (3,) — camera center in world coordinates.
			directions: np.array of shape (N, 3) — unit vectors in world frame.
		"""
		P_robot, R_world2robot = self._robot_pose(robot_pos, robot_orient)
		C_world, R_camera2world = self._camera_world_pose(P_robot, R_world2robot)

		pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)

		# Undistort to normalized coordinates on the ideal pinhole model
		# Returns Nx1x2 of normalized coordinates (x, y) on z=1 plane
		norm = cv2.undistortPoints(pts, self.K_camera_matrix, self.D_distorsion_coefficients)
		norm = norm.reshape(-1, 2)
		v_cam = np.column_stack([norm, np.ones(norm.shape[0])])  # (N,3)

		# Transform directions to world: v_world = R_cw @ v_cam
		v_world = (R_camera2world @ v_cam.T).T
		# Normalize to unit vectors
		v_world /= np.linalg.norm(v_world, axis=1, keepdims=True)

		origin = C_world.reshape(3)
		return origin, v_world

	# -------------------------------------------------------------------------------------
	# Helpers
	# -------------------------------------------------------------------------------------
	def _intrinsics_from_json(self, entry: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
		"""Build OpenCV intrinsic matrix K and distortion vector D from JSON entry.

		The JSON fields used:
			- width, height: image size in pixels
			- focal_x, focal_y: normalized by width (assumed); converted to pixels
			- c_x, c_y: principal point offsets normalized by width/height
			  (assumed relative to center). Converted to pixels via
			  cx = width/2 + c_x*width, cy = height/2 + c_y*height
			- k1, k2, p1, p2, k3: Brown distortion parameters.

		Returns:
			K: 3x3 intrinsic matrix
			D: 1x5 distortion vector (k1, k2, p1, p2, k3)
		"""
		w = int(entry.get("width"))
		h = int(entry.get("height"))
		max_dim = float(max(w, h))

		fx_n = float(entry.get("focal_x"))
		fy_n = float(entry.get("focal_y"))
		# Convert normalized focal to pixels using max_dim for consistency with the notebook
		fx = fx_n * max_dim
		fy = fy_n * max_dim

		cx_off = float(entry.get("c_x"))
		cy_off = float(entry.get("c_y"))
		# Principal point offsets also scaled by max_dim (matching the notebook exporter)
		cx = (w / 2.0) + cx_off * max_dim
		cy = (h / 2.0) + cy_off * max_dim

		K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

		D = np.array(
			[
				float(entry.get("k1", 0.0)),
				float(entry.get("k2", 0.0)),
				float(entry.get("p1", 0.0)),
				float(entry.get("p2", 0.0)),
				float(entry.get("k3", 0.0)),
			],
			dtype=np.float64,
		)
		return K, D
	
	def undistort_points(self, pixels, 
						 h_marker: float | None = None, 
						 h_camera: float | None = None,
						 h_plane: float | None = None,
						 perfect_camera: CameraModel | None = None) -> np.ndarray:
		"""
		Undistort pixel coordinates using the camera's intrinsic parameters.

		This method uses OpenCV's undistortPoints function to correct for lens distortion.

		Parameters:
		pixels (np.ndarray): An N x 2 array of pixel coordinates to undistort.
		h_marker (float, optional): The height of the marker above the floor, in meters. Defaults to 0. Is required for non-planarity correction.
		h_camera (float, optional): The height of the camera above the floor, in meters. Defaults to 0. Is required for non-planarity correction.
		h_plane (float, optional): The height of the reprojection plane above the floor, in meters. Defaults to 0. Is required for non-planarity correction.

		Returns:
		np.ndarray: An N x 2 array of undistorted pixel coordinates, with non-planarity correction if the heighs are provided.
		
		Raises:
		UserWarning: If some of the heights are passed but not all, in which case the non-planarity correction is not performed.
		"""
		# Convert to OpenCV format (N, 1, 2)
		pixel_dist = pixels.astype(np.float32).reshape(-1, 1, 2)

		if h_marker is not None and h_camera is not None and h_plane is not None:

			# Batch undistort using OpenCV (returns normalized coordinates)
			pixel_undist = cv2.undistortPoints(pixel_dist, self.K_camera_matrix, self.D_distorsion_coefficients, P=self.K_camera_matrix)
			# print(f"Pixel coordinates before non-planarity correction: {pixel_undist.reshape(-1, 2)} pixels")
			origin, rays = self.pixels_to_rays(np.zeros(2), np.zeros(1), pixel_undist.reshape(-1, 2))
			# Project the ray to relevant marker height, then to the plane height for non-planarity correction
			# cam_height = origin[2]
			marker_loc = intersect_plane(origin, rays[0], h_marker / 100.0)
			# print(f"Marker 3D location before non-planarity correction: {marker_loc}m")
			marker_loc[2] = h_plane / 100.0
			# print(f"Marker 3D location after non-planarity correction: {marker_loc}m")


			# Reproject 3D points back to pixel coordinates
			if perfect_camera is None:
				perfect_camera = self
			pixel_undist, _ = perfect_camera.project_points(
				robot_pos=np.zeros(2),
				robot_orient=np.zeros(1),
				points_3d=marker_loc.reshape(1, 3),
				return_distorted=False,
				return_visible=False,
			)
			pixel_undist = pixel_undist.reshape(-1, 2)

			# print(f"Pixel coordinates after non-planarity correction: {pixel_undist} pixels")

			# print(f"Non-planarity correction applied: marker at {h_marker}m, camera at {h_camera}m, plane at {h_plane}m → pixel shift of {(pixel_undist - pixels).flatten()} pixels")

		elif h_marker is not None or h_camera is not None or h_plane is not None:
			raise UserWarning("Only some of the h_marker, h_camera and h_plane passed. Skipping the plane reprojection.")
		else:
			# Batch undistort using OpenCV (returns normalized coordinates)
			pixel_undist = cv2.undistortPoints(pixel_dist, self.K_camera_matrix, self.D_distorsion_coefficients, P=self.K_camera_matrix)
			
			pixel_undist = pixel_undist.reshape(-1, 2)


		return pixel_undist

	def _rotation_matrix_from_euler(self, euler_xyz: np.ndarray) -> np.ndarray:
		"""Rotation matrix from Euler angles (rx, ry, rz) in radians.

		Uses Tait-Bryan angles with intrinsic rotations around X, then Y, then Z.
		The returned matrix rotates from the rotated frame to the original frame
		when used with column vectors.
		"""
		rx, ry, rz = np.asarray(euler_xyz, dtype=float).reshape(3)

		# Rotation about X
		Rx = np.array(
			[[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]],
			dtype=np.float64,
		)
		# Rotation about Y
		Ry = np.array(
			[[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]],
			dtype=np.float64,
		)
		# Rotation about Z
		Rz = np.array(
			[[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]],
			dtype=np.float64,
		)

		return Rz @ Ry @ Rx

	def _euler_component_matrices(self, euler_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Return Rx, Ry, Rz component matrices for given Euler angles (rx,ry,rz).

		Formulas:
			Rx = [[1,0,0],[0,cos(rx),-sin(rx)],[0,sin(rx),cos(rx)]]
			Ry = [[cos(ry),0,sin(ry)],[0,1,0],[-sin(ry),0,cos(ry)]]
			Rz = [[cos(rz),-sin(rz),0],[sin(rz),cos(rz),0],[0,0,1]]
		"""
		rx, ry, rz = np.asarray(euler_xyz, dtype=float).reshape(3)
		Rx = np.array(
			[[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]],
			dtype=np.float64,
		)
		Ry = np.array(
			[[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]],
			dtype=np.float64,
		)
		Rz = np.array(
			[[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]],
			dtype=np.float64,
		)
		return Rx, Ry, Rz

	def _euler_component_derivatives(self, euler_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Return dRx/drx, dRy/dry, dRz/drz matrices for given Euler angles.

		Formulas (element-wise derivatives):
			dRx/drx = [[0,0,0],[0,-sin(rx),-cos(rx)],[0,cos(rx),-sin(rx)]]
			dRy/dry = [[-sin(ry),0,cos(ry)],[0,0,0],[-cos(ry),0,-sin(ry)]]
			dRz/drz = [[-sin(rz),-cos(rz),0],[cos(rz),-sin(rz),0],[0,0,0]]
		"""
		rx, ry, rz = np.asarray(euler_xyz, dtype=float).reshape(3)
		dRx = np.array(
			[[0, 0, 0], [0, -np.sin(rx), -np.cos(rx)], [0, np.cos(rx), -np.sin(rx)]],
			dtype=np.float64,
		)
		dRy = np.array(
			[[-np.sin(ry), 0, np.cos(ry)], [0, 0, 0], [-np.cos(ry), 0, -np.sin(ry)]],
			dtype=np.float64,
		)
		dRz = np.array(
			[[-np.sin(rz), -np.cos(rz), 0], [np.cos(rz), -np.sin(rz), 0], [0, 0, 0]],
			dtype=np.float64,
		)
		return dRx, dRy, dRz

	def _robot_pose(self, robot_pos: np.ndarray, robot_orient: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""Normalize robot pose arrays and compute world rotation.

		- robot_pos: (2,) -> (x, y, 0); (3,) kept as-is
		- robot_orient: (1,) -> yaw (around +Z); (3,) -> (rx, ry, rz)
		"""
		position = np.asarray(robot_pos, dtype=float)
		if position.shape == (2,):
			position = np.array([position[0], position[1], 0.0], dtype=float)
		elif position.shape != (3,):
			raise ValueError("robot_pos must be shape (2,) or (3,)")

		orientation = np.asarray(robot_orient, dtype=float)
		if orientation.shape == (1,):
			yaw = float(orientation[0])
			R_wr = self._rotation_matrix_from_euler(np.array([0.0, 0.0, yaw]))
		elif orientation.shape == (3,):
			R_wr = self._rotation_matrix_from_euler(orientation)
		else:
			raise ValueError("robot_orient must be shape (1,) or (3,)")

		return position, R_wr

	def _camera_world_pose(self, P_robot: np.ndarray, R_world2robot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""Compute camera center and rotation in world frame.

		Definitions:
			C_world = P_robot + R_wr offsets, with R_wr = R_world2robot.
			R_cw = R_wr @ R_mount @ R_base is world_from_camera; R_wc = R_cw^T used for projection.

		Returns:
			C_world: camera center position (3,)
			R_cw: rotation (3x3) mapping camera-frame vectors to world-frame
		"""
		# Camera center: robot position + rotated offsets
		C_world = P_robot.reshape(3) + (R_world2robot @ self.offsets.reshape(3))

		# Camera rotation in world: world_from_camera
		R_cw = R_world2robot @ self.R_rot_matrix_mounting @ self.R_base
		return C_world.reshape(3), R_cw


def _load_default_json_path() -> str:
	"""Helper to resolve default camera.json next to this file."""
	here = os.path.dirname(os.path.abspath(__file__))
	return os.path.join(here, "../../../weights/top_camera/camera.json")


if __name__ == "__main__":
	# -------------------------------------------------------------------------------------
	# Minimal unit tests (vectorized behavior and basic geometry sanity)
	# -------------------------------------------------------------------------------------
	np.set_printoptions(precision=4, suppress=True)

	json_path = _load_default_json_path()
	assert os.path.exists(json_path), f"camera.json not found at {json_path}"

	# Camera mounted at robot origin with slight Z offset, no tilt (vertical)
	cam = CameraModel(offsets=(0.0, 0.0, 0.5), tilts_rad=(0.0, 0.0, 0.0), camera_json_path=json_path)

	# Test 1: Pixel center ray should point ~+Z in world when robot has zero yaw
	cx, cy = cam.K_camera_matrix[0, 2], cam.K_camera_matrix[1, 2]
	origin, dirs = cam.pixels_to_rays(
		robot_pos=np.array([0.0, 0.0]),
		robot_orient=np.array([0.0]),
		pixels=np.array([[cx, cy]]),
	)
	print("Test 1: Center ray sanity")
	print("  Expected origin z ≈ camera offset (0.5 m)")
	print("  Origin:", origin)
	print("  Expected direction ≈ [0, 0, 1]")
	print("  Direction:", dirs[0])
	assert np.allclose(dirs[0], np.array([0.0, 0.0, 1.0]), atol=5e-3), "Center pixel ray should align with +Z"

	# Test 2: Reciprocity — project world points to pixels, then back to rays
	# Construct points on a plane above the camera
	H = 3.0
	P_robot = np.array([0.0, 0.0])
	yaw0 = np.array([0.0])
	C, _ = cam.pixels_to_rays(P_robot, yaw0, np.array([[cx, cy]]))
	Zp = C[2] + H
	rng = np.random.default_rng(42)
	XY = rng.uniform(low=-1.0, high=1.0, size=(100, 2))
	P_world = np.column_stack([XY, np.full(100, Zp)])

	pixels, _ = cam.project_points(robot_pos=P_robot, robot_orient=yaw0, points_3d=P_world)
	print("Test 2: Reciprocity on random points above camera")
	print("  Projected pixels sample:", pixels[:5])
	assert pixels.shape == (P_world.shape[0], 2)
	# Filter only finite (in-frame) pixels
	mask = np.isfinite(pixels).all(axis=1)
	origin2, rays2 = cam.pixels_to_rays(robot_pos=P_robot, robot_orient=yaw0, pixels=pixels[mask])
	v_true = (P_world[mask] - origin2.reshape(1, 3))
	v_true /= np.linalg.norm(v_true, axis=1, keepdims=True)
	cos_sim = np.sum(v_true * rays2, axis=1)
	print("  Expected mean cos similarity ≈ 1.0")
	print("  Mean cos similarity:", np.mean(cos_sim))
	assert np.all(cos_sim > 1.0 - 2e-3), "Back-projected rays must align with true directions"

	# Test 3: Yaw case — rotate robot and verify reciprocity still holds
	yaw = np.deg2rad(15.0)
	pixels_yaw, _ = cam.project_points(robot_pos=P_robot, robot_orient=np.array([yaw]), points_3d=P_world)
	mask_y = np.isfinite(pixels_yaw).all(axis=1)
	origin3, rays3 = cam.pixels_to_rays(robot_pos=P_robot, robot_orient=np.array([yaw]), pixels=pixels_yaw[mask_y])
	v_true3 = (P_world[mask_y] - origin3.reshape(1, 3))
	v_true3 /= np.linalg.norm(v_true3, axis=1, keepdims=True)
	cos_sim3 = np.sum(v_true3 * rays3, axis=1)
	print("Test 3: Reciprocity under yaw=15°")
	print("  Expected mean cos similarity ≈ 1.0")
	print("  Mean cos similarity:", np.mean(cos_sim3))
	assert np.all(cos_sim3 > 1.0 - 2e-3), "Yawed back-projected rays must align with true directions"

	# Test 4: Principal point sanity — a point directly above camera projects near principal point
	P_above = origin.reshape(1, 3) + np.array([[0.0, 0.0, H]])
	px_above, _ = cam.project_points(robot_pos=P_robot, robot_orient=yaw0, points_3d=P_above)
	print("Test 4: Principal point sanity")
	print("  Expected projected pixel ≈ principal point")
	print("  Principal point:", (cx, cy))
	print("  Projection:", px_above[0])

	# Test 5: Corner rays and XY bounds across multiple heights
	print("Test 5: Corner rays and XY bounds across heights")
	corner_pixels = np.array([
		[0.0, 0.0],
		[cam.width - 1.0, 0.0],
		[0.0, cam.height - 1.0],
		[cam.width - 1.0, cam.height - 1.0],
	], dtype=np.float64)
	corner_origin, corner_dirs = cam.pixels_to_rays(robot_pos=P_robot, robot_orient=yaw0, pixels=corner_pixels)
	print("  Corner ray directions (z should be > 0):")
	for i, d in enumerate(corner_dirs):
		print(f"	Corner {corner_pixels[i]}: dir={d}")
	assert np.all(corner_dirs[:, 2] > 0.0), "Corner ray z-components must be positive (upward)"

	for H_test in [2.0, 3.0, 5.0]:
		z_plane = origin[2] + H_test
		corners_world = np.stack([intersect_plane(corner_origin, d, z_plane) for d in corner_dirs], axis=0)
		x_min, x_max = float(np.min(corners_world[:, 0])), float(np.max(corners_world[:, 0]))
		y_min, y_max = float(np.min(corners_world[:, 1])), float(np.max(corners_world[:, 1]))
		print(f"  Height {H_test:.1f} m: XY bounds from corner rays: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")
		# Sample points within bounds and test reciprocity
		rng2 = np.random.default_rng(123 + int(H_test * 10))
		XY_samp = np.column_stack([
			rng2.uniform(x_min, x_max, size=150),
			rng2.uniform(y_min, y_max, size=150),
		])
		P_world_test = np.column_stack([XY_samp, np.full(XY_samp.shape[0], z_plane)])
		px, _ = cam.project_points(robot_pos=P_robot, robot_orient=yaw0, points_3d=P_world_test)
		maskH = np.isfinite(px).all(axis=1)
		if not np.any(maskH):
			print(f"	Height {H_test:.1f} m: no finite projections within FoV; skipping ray check.")
			continue
		org, rays = cam.pixels_to_rays(robot_pos=P_robot, robot_orient=yaw0, pixels=px[maskH])
		v_trueH = (P_world_test[maskH] - org.reshape(1, 3))
		v_trueH /= np.linalg.norm(v_trueH, axis=1, keepdims=True)
		cosH = np.sum(v_trueH * rays, axis=1)
		print(f"	Height {H_test:.1f} m: mean cos similarity ≈ {np.mean(cosH):.6f}")
		assert np.all(cosH > 1.0 - 2e-3), "Corner-bounded reciprocity must hold across heights"
	assert np.linalg.norm(px_above[0] - np.array([cx, cy])) < 1.0, "Directly above should be near principal point"

	# Test 6: Jacobian sanity via finite differences on a single visible point
	print("Test 6: Jacobian sanity (finite differences)")
	P_robot_fd = np.array([0.2, -0.1, 0.0])
	angles_fd = np.array([np.deg2rad(1.0), np.deg2rad(-1.5), np.deg2rad(10.0)])
	H_fd = 3.0
	C_fd, _ = cam.pixels_to_rays(P_robot_fd[:2], np.array([angles_fd[2]]), np.array([[cx, cy]]))
	z_plane_fd = C_fd[2] + H_fd
	P_world_fd = np.array([[0.0, 0.0, z_plane_fd]])
	# Use full 3-angle orientation for Jacobian test
	pix_fd, J_fd = cam.project_points(robot_pos=P_robot_fd, robot_orient=angles_fd, points_3d=P_world_fd)
	assert np.isfinite(pix_fd).all(), "Finite projection required for Jacobian test"
	assert np.isfinite(J_fd).all(), "Jacobian must be finite for visible point"
	J_vis = J_fd.reshape(1, 2, 6)[0]
	# Central differences
	eps_pos = 1e-6
	eps_ang = 1e-6
	def proj(p, a):
		px, _ = cam.project_points(robot_pos=p, robot_orient=a, points_3d=P_world_fd)
		return px.reshape(2)
	base = proj(P_robot_fd, angles_fd)
	J_num = np.zeros((2, 6), dtype=np.float64)
	for i in range(3):
		p_plus = P_robot_fd.copy(); p_plus[i] += eps_pos
		p_minus = P_robot_fd.copy(); p_minus[i] -= eps_pos
		f_plus = proj(p_plus, angles_fd)
		f_minus = proj(p_minus, angles_fd)
		J_num[:, i] = (f_plus - f_minus) / (2 * eps_pos)
	for j in range(3):
		a_plus = angles_fd.copy(); a_plus[j] += eps_ang
		a_minus = angles_fd.copy(); a_minus[j] -= eps_ang
		f_plus = proj(P_robot_fd, a_plus)
		f_minus = proj(P_robot_fd, a_minus)
		J_num[:, 3 + j] = (f_plus - f_minus) / (2 * eps_ang)
	# Compare
	err = np.linalg.norm(J_vis - J_num) / (np.linalg.norm(J_num) + 1e-12)
	print("  Relative Jacobian error:", err)
	assert err < 5e-2, "Jacobian finite-difference check failed (rel error < 5%)"

	# Test 7: calibration for tilt angles: tilt the camer 10 degrees in 4 directions and verify the principal point ray tilts accordingly
	print("Test 7: Camera tilt sanity")
	tilt_angles = [
		(np.deg2rad(10.0), 0.0, np.deg2rad(90.0)),   # 10° pitch
		(np.deg2rad(-10.0), 0.0, np.deg2rad(90.0)),  # -10° pitch
		(0.0, np.deg2rad(10.0), np.deg2rad(90.0)),   # 10° roll
		(0.0, np.deg2rad(-10.0), np.deg2rad(90.0)),  # -10° roll
	]
	for tilt in tilt_angles:
		cam_tilted = CameraModel(offsets=(0.0, 0.0, 0.5), tilts_rad=tilt, camera_json_path=json_path)
		origin_tilt, dir_tilt = cam_tilted.pixels_to_rays(
			robot_pos=np.array([0.0, 0.0]),
			robot_orient=np.array([0.0]),
			pixels=np.array([[cx, cy]]),
		)
		print(f"  Tilt (rx,ry,rz)={tilt}: direction={dir_tilt[0]}")
		# Check that the z-component is less than 1.0 (i.e., tilted away from +Z)
		assert dir_tilt[0][2] < 1.0, "Tilted camera center ray must deviate from +Z"

	print("All tests passed.")

