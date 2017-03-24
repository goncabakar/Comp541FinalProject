
  for p in ("Knet","ArgParse","ImageMagick","MAT","Images")
      Pkg.installed(p) == nothing && Pkg.add(p)
  end

  using Knet,ArgParse,Images,MAT

  function main(args="")
      batchsize = 100
      for i=1:2020
      xtrn = loaddata("faces", batchsize, i)
      #Because my rem is insufficient,
      #i load images from current minibatch only.
      #Total number of images is 202000
      end
      end


  function loaddata(dataset, bs, n)
  if dataset == "faces"
  path = "/Users/goncabakar/Desktop/hop/4/machine_learning/facesdataset/img_align_celeba_png/"
  e1 = zeros(Float32,(218, 178, 3, bs))
  index = 1
  for i=(n-1)*bs+1:n*bs
    str = "$i"
    if (sizeof(str) == 1)
      str = path * "00000" * str * ".png"
    elseif sizeof(str) == 2
      str = path * "0000" * str * ".png"
    elseif sizeof(str) == 3
      str = path * "000" * str * ".png"
    elseif sizeof(str) == 4
      str = path * "00" * str * ".png"
    elseif sizeof(str) == 5
      str = path * "0" * str * ".png"
    else sizeof(str) == 6
      str = path * str * ".png"
    end
    a0 = load(str)
    a1 = channelview(a0)
    b1 = convert(Array{Float32}, a1)
    c1 = permutedims(b1, (3,2,1))
    d1 = reshape(c1[:,:,1:3], (178,218,3,1))
    e1[:,:,:,index] = permutedims(d1, [2,1,3,4])
    println(str)
    println(index)
    index = index + 1
  end
  return e1
  end
  end

  function cnn(xtrn)
    prob = randn(100, 1)
    return prob
  end

  function dcgan()
    bs = 10
    fake_images = rand(178, 218, 3, bs)
    return fake_images
  end
main()
