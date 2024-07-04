function pbc_displacement(xi::Vector,xj::Vector,L::Float64)
  xij = xi - xj
  xij_pbc = map(xu -> argmin(abs, [(xu), (xu.+L), (xu.-L)]), xij)
  return xij_pbc
end

pbc_distance(xi::Vector,xj::Vector,L::Float64) = norm(pbc_displacement(xi,xj,L))

function perp_diff_spin(Sia::Vector,q::Vector)
  qhat = normalize(q) 
  return Sia - qhat * dot(qhat,Sia)
end

function single_msf(centers::Matrix,dirs::Matrix,rels::Matrix,N::Int64,a::Float64,q::Vector)
  cutoff = 10*a 
  suma = 0 # initialize

  # loop through all pairs
  for i in 1:size(centers,1), j in i:size(centers,1)
    
    riajb = pbc_displacement(centers[i,:], centers[j,:], N*a)
    if norm(riajb) <= cutoff
      Sia = perp_diff_spin(dirs[i,1:2],q)
      Sjb = perp_diff_spin(dirs[j,1:2],q)

      term = dot(Sia,Sjb) * cis( dot(q,riajb[1:2]) )
      suma += real(term)
    else
        continue
    end

  end
  
  return suma/2/N^2
end

function trj2array(trj::DataFrame)
    centers = Matrix(select( trj, [:x,:y,:z] ) ) 
    dirs = Matrix(select( trj, [:dx,:dy,:dz] ) ) 
    rels = Matrix(select( trj, [:cx,:cy,:cz] ) )
    return centers, dirs, rels
end


function reciprocal_space(a;amount_bz = 1)
  bz = pi/a * amount_bz 
  kx = range(-bz,bz,length=120)
  ky = range(-bz, bz, length=120)

  mesh = [[kxi,kyi] for kyi in ky, kxi in kx]
  return kx,ky,mesh
end
