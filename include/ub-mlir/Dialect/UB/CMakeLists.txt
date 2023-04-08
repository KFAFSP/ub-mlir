################################################################################
# UBIncGen
#
# MLIR UB dialect generated includes.
################################################################################

add_custom_target(UBIncGen)

# Interfaces.
add_subdirectory(Interfaces)
# Attributes, Dialect, Operations and Types.
add_subdirectory(IR)