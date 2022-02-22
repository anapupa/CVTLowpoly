from dis import dis
import numpy as np
import taichi as ti
import taichi_glsl as ts
from .jumping_flooding import jfa_solver_2D


ti.init()


@ti.data_oriented
class TriRaster:
    def __init__(self, width, height, points, triangle):
        # (x,y) denotes the coordinates of centroid
        # z is the auxiliary component to record the number of pixel in current voronoi region
        self.num_points = len(points)
        self.num_tri    = len(triangle)
        self.w, self.h = width, height
        points[:, 0] *= width
        points[:, 1] *= height
        self.points = ti.Vector.field(points.shape[1], dtype=ti.f32, shape=points.shape[0])
        self.points.from_numpy(points)

        self.color = ti.Vector.field(3, dtype=ti.f32,  shape=triangle.shape[0])
        self.color_w = ti.Vector.field(1, dtype=ti.f32,  shape=triangle.shape[0])
        self.color.fill(0)
        self.color_w.fill(0)

        # self.tris = ti.Vector.field(3, dtype=ti.i32, shape=triangle.shape[-1])
        self.tris = ti.Vector.field(triangle.shape[1], dtype=ti.i32, shape=triangle.shape[0])
        self.tris.from_numpy(triangle)

        self.pixels = ti.field(ti.i32, shape=(width, height))
        self.pixels.fill(-1)


    @ti.func
    def edge_left_test(self, a: ti.template(), b: ti.template(), c: ti.template()):
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]) 

    @ti.kernel 
    def fill_triangle_with_id(self):
        for i in range(self.num_tri):
            # print(i, self.tris[i, 0], self.tris[i, 1], self.tris[i, 2])
            a, b, c = self.points[self.tris[i].x], self.points[self.tris[i].y], self.points[self.tris[i].z]
            X, Y = ts.vec3(a.x, b.x, c.x), ts.vec3(a.y, b.y, c.y)

            area = self.edge_left_test(a, b, c)

            xrange = (int( ts.floor(ts.minimum(X)) ), int( ts.ceil(ts.maximum(X)+1 ) ) )
            yrange = (int( ts.floor(ts.minimum(Y)) ), int( ts.ceil(ts.maximum(Y)+1 ) ) )

            # # print(xrange, yrange)
            for x, y in ti.ndrange( xrange, yrange ):
                p = ts.vec2(x+0.5, y+0.5)
                w0 = self.edge_left_test(b, c, p)  
                w1 = self.edge_left_test(c, a, p)  
                w2 = self.edge_left_test(a, b, p)  
                if w0 <= 0 and w1 <= 0 and w2 <= 0 :
                    self.pixels[x, y] = i


    def solve(self):
        self.fill_triangle_with_id()


    @ ti.kernel
    def render_color(self, screen: ti.template(), site_info: ti.template()):
        for I in ti.grouped(screen):
            if self.pixels[I] != -1:
                screen[I] = site_info[self.pixels[I]]
            else:
                screen[I].fill(-1)

    @ti.kernel
    def render_index(self, screen: ti.template()):
        for I in ti.grouped(screen):
            if self.pixels[I] != -1:
                screen[I].fill(self.pixels[I] / self.num_tri)
            else:
                screen[I].fill(-1)

    @ti.kernel
    def compute_mean_color(self, image: ti.template()): # , density: ti.template()
        for i, j in self.pixels:
            index = self.pixels[i, j]
            if index == -1: continue
            # dist = density[j, i]
            dist = ts.distance(ts.vec(i/self.w, j/self.h), self.points[self.pixels[i, j]])
            # dist = 1.0
            self.color[index].x += image[j, i].x * dist
            self.color[index].y += image[j, i].y * dist
            self.color[index].z += image[j, i].z * dist
            self.color_w[index] += dist 

        for i in range(self.num_tri):
            # if self.centroids[i].z > 0:
            if  self.color_w[i].x == 0:
                p = self.points[i]
                self.color[i] = image[int(p.y), int(p.x)]
            else:
                self.color[i].x /= self.color_w[i].x
                self.color[i].y /= self.color_w[i].x
                self.color[i].z /= self.color_w[i].x

    def get_triangle_mean_color(self, image: np.ndarray):
        ti_image = ti.Vector.field(3, dtype=ti.f32, shape=(self.h, self.w))
        ti_image.from_numpy(image)
        self.compute_mean_color(ti_image)

        # print(self.color.to_numpy())
        return self.color.to_numpy()
