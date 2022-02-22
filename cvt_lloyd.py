import re
import numpy as np
import taichi as ti
import taichi_glsl as ts
from .jumping_flooding import jfa_solver_2D


ti.init()


@ti.data_oriented
class cvt_lloyd_solver_2D:
    def __init__(self, width, height, init_sites, density = None):
        # (x,y) denotes the coordinates of centroid
        # z is the auxiliary component to record the number of pixel in current voronoi region
        self.centroids = ti.Vector.field(3, dtype=ti.f32,  shape=init_sites.shape[0])
        centroids = np.column_stack([ init_sites, np.ones(len(init_sites)) ]).astype(np.float32)
        # print(init_sites.shape, centroids.shape)
        self.centroids.from_numpy(centroids)
        if type(density) == type(None): density = np.ones((width, height)).astype(np.float32)

        self.density = ti.field(ti.f32, shape=(width, height))
        self.density.from_numpy(density.T.astype(np.float32))
        

        # since jfa_solver will use from_numpy()
        # it must be put after all taichi variables
        self.jfa = jfa_solver_2D(width, height, init_sites)

    def solve_cvt(self, m=5):
        step_x = int(np.power(2, np.ceil(np.log(self.jfa.w))))
        step_y = int(np.power(2, np.ceil(np.log(self.jfa.h))))
        iteration = 0
        while True:
            self.jfa.solve_jfa((step_x, step_y))
            self.compute_centroids()
            if self.cvt_convergence_check() == 1:
                break
            self.jfa.assign_sites(self.centroids)
            # Using 2 * maximum average distance as the jfa step for the first m iteration
            # if iteration <= m:
            #     pass
            iteration += 1
            if iteration > m: break
        # print("iteration times:", iteration)

    @ti.kernel
    def compute_centroids(self):
        for i in range(self.jfa.num_site):
            self.centroids[i].x *= 0.01
            self.centroids[i].y *= 0.01
            self.centroids[i].z = 0.01
        for i, j in self.jfa.pixels:
            index = self.jfa.pixels[i, j]
            w = self.density[i, j]
            self.centroids[index].x += i / self.jfa.w * w 
            self.centroids[index].y += j / self.jfa.h * w
            self.centroids[index].z += w
        for i in range(self.jfa.num_site):
            # if self.centroids[i].z > 0: 
            self.centroids[i].x /= self.centroids[i].z
            self.centroids[i].y /= self.centroids[i].z

    @ti.kernel
    def cvt_convergence_check(self) -> ti.i32:
        end_flag = 1
        for i in range(self.jfa.num_site):
            dist = ts.distance(self.jfa.sites[i], ts.vec(
                self.centroids[i].x, self.centroids[i].y))
            if dist > 0:
                end_flag = 0
        return end_flag
