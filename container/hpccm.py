import hpccm
Stage0 = hpccm.Stage()
Stage0 += hpccm.primitives.baseimage(image='nvidia/cuda:10.2-devel-centos7')
Stage0 += hpccm.building_blocks.mlnx_ofed(version='5.0-2.1.8.0')
compiler = hpccm.building_blocks.gnu()
Stage0 += hpccm.building_blocks.fftw(version='3.3.8', mpi=True, toolchain=compiler.toolchain)
Stage0 += hpccm.building_blocks.hdf5(version='1.10.5', toolchain=compiler.toolchain)
Stage0 += hpccm.building_blocks.openmpi(cuda=True, infiniband=True)
print(Stage0)
