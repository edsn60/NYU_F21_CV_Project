batch_size = 64
momentum = 0.9
lr = 0.01
optimizer = 'adam'
epochs = 30

out_size_large = 8
out_size_small = 22


train = False
saved_model_large_path = './saved_models/FKP_model_large.pt'
saved_model_small_path = './saved_models/FKP_model_small.pt'

image_size = 96

valid_partition = 0.2

datasets_tags = {'large_kpset': ['left_eye_center_x', 'left_eye_center_y',
                                  'right_eye_center_x', 'right_eye_center_y',
                                  'nose_tip_x', 'nose_tip_y',
                                  'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y',
                                  'Image'
                                 ],
                 'small_kpset': [ 'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
                                  'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
                                  'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
                                  'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
                                  'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
                                  'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
                                  'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
                                  'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
                                  'mouth_left_corner_x', 'mouth_left_corner_y',
                                  'mouth_right_corner_x', 'mouth_right_corner_y',
                                  'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
                                  'Image'
                                 ]
                }