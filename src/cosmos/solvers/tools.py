from collections import Counter

def print_mesh_info(mesh):
    """_summary_
    """

    print(60*'-')

    print('The mesh is a ', str(mesh.dim) + 'D mesh')

    if mesh.ne == 0:
        print('! The mesh is also an hypersurface embedded in ', str(mesh.dim) + 'D !')

    print('The number of:')
    print(5*' '+'vertices is: ', mesh.nv)
    print(5*' '+'edges is: ', mesh.nedge)
    print(5*' '+'faces is: ', mesh.nface)
    print(5*' '+'facets is: ', mesh.nfacet)
    print(5*' '+'elements is: ', mesh.ne, '( = 0 if hypersurface)')

    regions = [mesh.GetMaterials(),
                mesh.GetBoundaries(), 
                mesh.GetBBoundaries(), 
                mesh.GetBBBoundaries()]

    for i in range(mesh.dim+1):

        if regions[i]:

            cnt = Counter(regions[i])

            print('Regions of co-dimension ', i,':')
            for key, value in cnt.items():
                print(' N. ', value, 'region(s) called ', key)

    print(60*'-', '\n')