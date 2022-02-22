from random import sample
import cv2, time
import numpy as np
from scipy.spatial import Delaunay

from .cvt_lloyd import cvt_lloyd_solver_2D
from .tri_raster import TriRaster
import taichi as ti 

ti.init(arch=ti.gpu, kernel_profiler=True)


def local_linear_coefficient(p, I, r, epsilon):
    p, I = np.copy(p).astype(np.float32), np.copy(I).astype(np.float32)
    I_mean  = cv2.boxFilter(I, -1, (r,r) )
    p_mean = cv2.boxFilter(p, -1, (r,r))
    Ip_mean = cv2.boxFilter(I*p, -1, (r,r))
    II_mean = cv2.boxFilter(I*I, -1, (r,r))

    covIp = Ip_mean - I_mean * p_mean    
    varI = II_mean - I_mean * I_mean
    
    a = covIp/(varI + epsilon)
    b = p_mean - a * I_mean

    a_mean = cv2.boxFilter(a, -1, (r,r))
    b_mean = cv2.boxFilter(b, -1, (r,r))

    return (a * I + b_mean * 0.1).astype(np.uint8)

def edge_feature(image: np.ndarray):
    feature = np.zeros(image.shape[:2])
    for img in cv2.split(image):
        smoothed = cv2.bilateralFilter(img, 7, 20, 100)
        # smoothed = img
        gradient = local_linear_coefficient(img, smoothed, 10, (0.03*255) ** 2)
        # gradient[np.where(gradient < 100)] =  0
        gradient = local_linear_coefficient(gradient, smoothed, 3, (0.15*255) ** 2)
        gradient = gradient / np.max(gradient)
        gradient[np.where( gradient < 0.01)] += 0.01
        feature = np.maximum(feature, gradient)
    return feature

def feature_sampling(feature: np.ndarray, n_samples: np.int32):
    density = np.copy(feature.flatten() )
    ratio = min(np.mean(density)*5, 0.5)
    density1 = np.minimum(density, 0.2)
    density1[np.where(density1 < 0.1)] = 0

    # cv2.imshow("sharp edge", density1.reshape(feature.shape))
    # cv2.waitKey(0)   

    samples_flat  = np.random.choice(density.size, int(n_samples * (1-ratio) ) ) 
    samples_sharp = np.random.choice(density.size, int(n_samples * ratio ), p = density1 / np.sum(density1) ) 
    samples = np.unique(np.concatenate([samples_flat, samples_sharp])) 
    # print(density.size, samples.size )
    # samples = np.random.choice(feature.size, int(n_samples), p = density / np.sum(density) ) 
    y = np.floor(samples/feature.shape[1]) / feature.shape[0]
    x = np.mod(samples, feature.shape[1]) / feature.shape[1]
    # img = np.zeros_like(feature)
    # for i in range(len(x)):
    #     img = cv2.circle(img, (int(x[i]*feature.shape[1]), int(y[i]*feature.shape[0]) ), 1, (255, 0, 0) ) 
    # cv2.imshow("image", img)
    # cv2.waitKey(0)   
    xy = np.column_stack([x, y])
    # print(np.unique(xy, axis=1).shape, xy.shape)
    return xy 

def lowpoly_mesh(I: np.ndarray): 
    feature = edge_feature(I)
    seeds = feature_sampling(feature, feature.size * 0.01).astype(np.float32)
    feature = np.maximum(feature**2, 0.01)
    # feature = np.ones(image.shape[:2])
    # seeds =  np.array(np.random.rand(int(feature.size * 0.01), 2), dtype=np.float32)

    h, w = feature.shape
    # print("h, w", feature.shape)
    step = (int(np.power(2, np.ceil(np.log(w)))),
            int(np.power(2, np.ceil(np.log(h)))))
    screen = ti.Vector.field(3, dtype=ti.f32, shape=(w, h))
    # seeds = np.array(np.random.rand(len(seeds), 2), dtype=np.float32)
    seeds_info = np.array(np.random.rand(len(seeds)*2, 3), dtype=np.float32)
    info = ti.Vector.field(3, dtype=ti.f32, shape=seeds_info.shape[0])
    cvt_solver = cvt_lloyd_solver_2D(w, h, seeds, feature)
    info.from_numpy(seeds_info)

    # cvt_solver.jfa.solve_jfa(step)
    # cvt_solver.jfa.render_color(screen, info)
    # ti.imwrite(screen.to_numpy(), './outputs/jfa_output.png')

    cvt_solver.solve_cvt(5)
    final_seeds = cvt_solver.centroids.to_numpy()[:, :2]

    # print("time cost: ",time.time() - begin_time)
    # cvt_solver.jfa.render_color(screen, info)
    # ti.imwrite(screen.to_numpy(), './outputs/cvt_output.png')

    # init_img, img = np.zeros_like(feature), np.zeros_like(feature)
    # for i in range(len(seeds)):
    #     init_img = cv2.circle(init_img, (int(seeds[i][0]*feature.shape[1]), int(seeds[i][1]*feature.shape[0]) ), 1, (255, 0, 0) ) 
    #     img = cv2.circle(img, (int(final_seeds[i][0]*feature.shape[1]), int(final_seeds[i][1]*feature.shape[0]) ), 1, (255, 0, 0) ) 
    # cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    # # cv2.imwrite("media/output/1.png", np.concatenate([init_img, img], axis=1))
    # cv2.imshow("image",np.concatenate([feature, init_img, img ], axis=1))
    # cv2.waitKey(0)  

    # print(image.shape, screen.to_numpy().shape) 

    x, y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
    grid_p = np.column_stack([x.flatten(), y.flatten()])

    tri = Delaunay( np.row_stack( [final_seeds,grid_p])  )
    return tri.points, tri.simplices, feature

def lowpoly_image(I: np.ndarray):
    points, triangles, feature = lowpoly_mesh(I)
    h, w = I.shape[:2]
    raster = TriRaster(w, h, points.astype(np.float32), triangles)
    raster.solve()
    # density = ti.Vector.field(3, dtype=ti.f32, shape=seeds_info.shape[0])
    colors = raster.get_triangle_mean_color(I.astype(np.float32)/255 if np.max(I) > 1.0 else I)

    # raster.render_index(screen)
    screen = ti.Vector.field(3, dtype=ti.f32, shape=(w, h))
    raster.render_color(screen, raster.color)
    final_lowpoly = np.swapaxes(screen.to_numpy(), 0, 1)*255
    return final_lowpoly.astype(np.uint8), points, triangles, colors 

if __name__ == '__main__':
    for i in range(1, 6):
        # i = 5
        image = cv2.imread('./media/'+str(i)+'.jpeg', cv2.IMREAD_ANYCOLOR)
        begin_time = time.time()
        final_lowpoly, _, _, _ = lowpoly_image(image)
        print("lowpoly time cost: ", time.time() - begin_time)
        cv2.imwrite("media/output/lowpoly-"+str(i)+'.png', final_lowpoly)
        cv2.imshow("triangles", final_lowpoly )
        cv2.waitKey()
