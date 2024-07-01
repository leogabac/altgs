function trj2array(trj::DataFrame)
    centers = Matrix(select( trj, [:x,:y,:z] ) ) 
    dirs = Matrix(select( trj, [:dx,:dy,:dz] ) ) 
    rels = Matrix(select( trj, [:cx,:cy,:cz] ) )
    return centers, dirs, rels
end


function vrt_lattice(a::Float64,N::Int64)
    x = 0:a:(N-1)*a
    y = 0:a:(N-1)*a 

    X = [xi for yi in y, xi in x]
    Y = [yi for yi in y, xi in x]

    lattice = zeros(N,N,3)
    lattice[:,:,1] = X
    lattice[:,:,2] = Y

    return lattice 
end


function fix_position(position::Vector,a::Float64,N::Int64)

    L = N*a

    # Apply BC to X
    position[1] = position[1] % L
    if position[1] <0
        position[1] += L
    end

    # Apply BC to y
    position[2] = position[2] % L
    if position[2] <0
        position[2] += L
    end

    return position
end


function get_idx_from_position(centers::Matrix,pos::Vector,tol=0.1)

    for (i,center) in enumerate(eachrow(centers))
        distance = norm(center - pos)
        if isapprox(distance,0,atol=tol)
            return i 
        end
    end
    
end


function indices_lattice(vrt_space::AbstractArray,centers::Matrix,a::Float64,N::Int64)

    rows, cols = size(vrt_space)[1:2]
    indices_matrix = zeros(Int64,rows,cols,4)


    for i in 1:rows, j in 1:cols

        cur_vrt = vrt_space[i,j,:]

        # get the positions with pbc
        up = fix_position(cur_vrt + [0,a/2,0],a,N)
        down = fix_position(cur_vrt + [0,-a/2,0],a,N)
        left = fix_position(cur_vrt + [-a/2,0,0],a,N)
        right = fix_position(cur_vrt + [a/2,0,0],a,N)

        # get the indices_lattice
        up_idx = get_idx_from_position(centers,up)
        down_idx = get_idx_from_position(centers,down)
        left_idx = get_idx_from_position(centers,left)
        right_idx = get_idx_from_position(centers,right)

        indices_matrix[i,j,:] = [up_idx,down_idx,left_idx,right_idx]

    end

    return indices_matrix
    
end


function dipole_lattice(centers::Matrix,dirs::Matrix,rels::Matrix,vrt_space::AbstractArray,indices_matrix::AbstractArray)

    rows,cols = size(vrt_space)[1:2]
    arrow_lattice = zeros(rows,cols,3)

    for i in 1:rows, j in 1:cols

        cidxs = indices_matrix[i,j,:]
        arrow_direction = sum(dirs[cidxs,:], dims=1) |> vec |> normalize
        arrow_lattice[i,j,:] = arrow_direction

    end
    return arrow_lattice
end