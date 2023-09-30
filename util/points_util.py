def savePlyFromPtsRGB(pts, rgb, file_name, alpha=None):
    with open(file_name, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % pts.shape[0])
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        if alpha is not None:
            f.write('property uchar alpha\n')
        f.write('end_header\n')
        # write data
        for i in range(pts.shape[0]):
            if alpha is None:
                f.write('%f %f %f %d %d %d\n' % (pts[i,0], pts[i,1], pts[i,2], rgb[i,0]*255, rgb[i,1]*255, rgb[i,2]*255))
            else:
                f.write('%f %f %f %d %d %d %d\n' % (pts[i,0], pts[i,1], pts[i,2], rgb[i,0]*255, rgb[i,1]*255, rgb[i,2]*255, alpha[i]*255))