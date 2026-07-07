SIMULATED_OPERATIONS = {
    "eye_resize": "Shrink/Enlarge eyes",
    "eye_distance": "Decrease/Increase eye distance",
    "nose_bridge": "Narrow/Widen nose bridge",
    "nose_alar": "Narrow/Widen nasal alae",
    "nose_length": "Shorten/Lengthen nose",
    "mouth_position": "Move mouth downward/upward",
    "lip_thickness": "Thin/Plump lips",
    "mouth_resize": "Shrink/Enlarge mouth",
}

PROFESSIONAL_OPERATION_GROUPS = {
    "face_shape": [
        ["face_trans"],
        ["jaw_trans"],
        ["mandible_left", "mandible_right"],
        ["temple_left", "temple_right"],
    ],
    "eyes_brows": [
        ["eyebrow_distance_left", "eyebrow_distance_right"],
        ["eyebrow_height_left", "eyebrow_height_right"],
        ["eyebrow_size_left", "eyebrow_size_right"],
        ["eye_distance_left", "eye_distance_right"],
        ["eye_height_left", "eye_height_right"],
        ["eye_width_left", "eye_width_right"],
        ["eye_trans_left", "eye_trans_right"],
        ["eye_up_down_left", "eye_up_down_right"],
    ],
    "nose": [
        ["scale_nose"],
        ["shrink_nose"],
        ["nasal_root"],
        ["nasal_tip"],
        ["nose_longer"],
    ],
    "mouth": [
        ["high_mouth"],
        ["mouth_high"],
        ["mouth_smile"],
        ["mouth_breadth"],
        ["mouth_trans"],
    ],
    "body": [
        ["body_shape_right_shoulder"],
        ["body_shape_thin_shoulders"],
        ["body_shape_slim_hand"],
        ["body_shape_slim_leg"],
        ["body_shape_slim_waist"],
    ],
}
