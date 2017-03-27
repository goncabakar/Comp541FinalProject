
  for p in ("Knet","ArgParse","ImageMagick","MAT","Images")
      Pkg.installed(p) == nothing && Pkg.add(p)
  end

  using Knet,ArgParse,Images,MAT

  function main(args="")
      batchsize = 100
      for i=1:2020
      xtrn = loaddata("faces", batchsize, i)
      w_gen = weights_generator()
      w_disc = weights_discriminator
      @time for epoch=1:3
          train(w_gen, w_disc, xtrn)
      end
      #Because my ram is insufficient,
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
    #println(str)
    #println(index)
    index = index + 1
  end
  return e1
  end
  end


  function generator(w_gen, noise)
    #use deconvolution layers to generate image from noise
    #use deconv4 for deconvolution operator
    return fake_images
  end

  function discriminator(w_disc, image)
    #decides whether image is fake or real
    #inputs can be generated images or real images from dataset
  end

  function leakly_relu(x)
  return  max(0.01*x,x)
  end

  function weights_discriminator()
    #initalize weights for discriminator
    #it is a binary classifier
  end
  function weights_generator()
    #initalize weights for generator
  end

  function loss_discriminator(w_disc, w_gen, real_image, noise)
  #use for update discriminator's weights
  ypred_real = discriminator(w_disc,real_image)
  ynorm_real = logp(ypred_real,1)
  J_real = -sum(ygold_true .* ynorm_real) / size(ynorm_real,2)
  fake_image = generator(w_gen, noise)
  ypred_fake = discriminator(w_disc,fake_image)
  ynorm_fake = logp(1 - ypred_fake,1)
  J_fake = -sum(ygold_false .* ynorm_fake) / size(ynorm_fake,2)
  return J_real + J_fake
  end

  function loss_generator(w_gen, w_disc, noise)
  #use for update generator's weights
  fake_image = generator(w_gen, noise)
  ypred = discriminator(w_disc,fake_image)
  ynorm = logp(ypred,1)
  J = -sum(ygold .* ynorm) / size(ygold,2)
  return J
  end

  grad_gen = grad(loss_generator)
  grad_disc = grad(loss_discriminator)

  function train(w_gen, w_disc, xtrn)
  noise = randn(100)
  gradient_gen = grad_gen(w_gen, w_disc, noise)
  gradient_disc = loss_discriminator(w_disc, w_gen, xtrn, noise)
  opts_gen = map(x->Adam(), w_gen)
  for i in 1:length(w_gen)
      update!(w_gen, gradient_gen, opts_gen)
  end
  opts_disc = map(x->Adam(), w_disc)
  for i in 1:length(w_dics)
      update!(w_dics, gradient_dics, opts_disc)
  end

  return w_gen, w_disc
  end

main()
