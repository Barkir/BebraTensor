import creators.small_model as small_model
import creators.small_model2 as small_model2
import creators.ten_models as ten_models

small_model.create()
small_model2.create_arithmetic_model()
ten_models.create_reshape_matmul()
ten_models.create_conv_bn_relu()
ten_models.create_complex_elementwise()
ten_models.create_concat_split()
ten_models.create_conv_bn_relu()
ten_models.create_depthwise_separable_conv()
ten_models.create_gemm_classifier()
ten_models.create_pad_convolution()
ten_models.create_pooling_stages()
ten_models.create_residual_block()
