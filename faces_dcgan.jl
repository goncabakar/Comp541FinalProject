
  for p in ("Knet","ArgParse","ImageMagick","MAT","Images")
      Pkg.installed(p) == nothing && Pkg.add(p)
  end

  using Knet,ArgParse,ImageMagick,Images,MAT

  function main(args="")
      batchsize = 128
      xtst = Any[]
      xtst0 = Any[]
      xtst0 = loaddata("faces", batchsize, 1201)
      push!(xtst, xtst0)
      xtst0= loaddata("faces", batchsize, 1202)
      push!(xtst, xtst0)
      xtst0 = loaddata("faces", batchsize, 1203)
      push!(xtst, xtst0)
      w_gen = weights_generator()
      w_disc = weights_discriminator()
      opts_disc = map(x->Adam(lr=0.0002, beta1=0.5), w_disc)
      opts_gen = map(x->Adam(lr=0.0002, beta1=0.5), w_gen)

      noise = randn(Float32,100,batchsize)
      #noise = convert(KnetArray{Float32}, noise)
      fake_image = generator(w_gen, noise; mode=0)
      loss_gen = loss_generator(w_gen, w_disc, noise)
      loss_disc = loss_discriminator(w_disc, loaddata("faces", batchsize, 1), fake_image)

      report(0,loss_gen,loss_disc,accuracy(w_gen, w_disc, xtst,0))
      @time for epoch=1:10
        for i=1:1
      #for i=1:1200
      xtrn = loaddata("faces", batchsize, i)
      train(w_gen, w_disc, xtrn, epoch, xtst, opts_disc, opts_gen)
      end
      end

  end

  report(epoch, lossgen, lossdisc, accuracy)=println((:epoch,epoch,:generatorloss,lossgen,:discriminatorloss,lossdisc,:accuracy_real_fake,accuracy))


  function loaddata(dataset, bs, n)
  if dataset == "faces"
  path = "/Users/goncabakar/Desktop/hop/4/machine_learning/facesdataset/img_align_celeba_png/"
  #e1 = zeros(Float32,(218, 178, 3, bs))
  e2 = zeros(Float32,(176, 176, 3, bs))
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
    d1 = reshape(c1[2:177,22:197,1:3], (176,176,3,1))
    e2[:,:,:,index] = permutedims(d1, [2,1,3,4])
    index = index + 1
  end
  e2 = 2*(e2.-0.5)
  #return map(a->convert(KnetArray{Float32},a), e2)
  return e2
  end
  end

  function generator(w, noise; mode=0)
    #use deconvolution layers to generate image from noise
    #use deconv4 for deconvolution operator
    bs = size(noise,2);
    x = w[1]*noise .+ w[2];
    x = reshape(x, 11, 11, 512, bs);
    x = batchnorm(x);
    x = relu(x);
    x = batchnorm(deconv4(w[3],x;padding=2, stride=2) .+ w[4]);
    x = relu(x);
    x = batchnorm(deconv4(w[5],x;padding=2, stride=2) .+ w[6]);
    x = relu(x);
    x = batchnorm(deconv4(w[7],x;padding=2, stride=2) .+ w[8]);
    x = relu(x);
    #x = batchnorm(deconv4(w[9],x;padding=2, stride=2) .+ w[10]);
    x = (deconv4(w[9],x;padding=2, stride=2) .+ w[10]);
    if mode == 1
      fake_images = sigm(x);
    else
      fake_images = tanh(x);
    end
    return fake_images
  end

  function discriminator(w, x)
    #decides whether image is fake or real
    #inputs can be generated images or real images from dataset

    #x = batchnorm(conv4(w[1],x;padding=2, stride=2) .+ w[2])
    x = (conv4(w[1],x;padding=2, stride=2) .+ w[2])
    x = leaky_relu(x)

    x = batchnorm(conv4(w[3],x;padding=2, stride=2) .+ w[4])
    x = leaky_relu(x)

    x = batchnorm(conv4(w[5],x;padding=2, stride=2) .+ w[6])
    x = leaky_relu(x)

    x = batchnorm(conv4(w[7],x;padding=2, stride=2) .+ w[8])

    x = mat(x)
    x = w[9]*x .+ w[10]
    x = sigm(x)

    return x
  end

  function leaky_relu(x)
  return  max(0.2*x,x)
  end

  function weights_discriminator()
    #initalize weights for discriminator
    #it is a binary classifier
    winit = 0.02
    w = Array(Any,10)
    w[1] = winit*randn(Float32, 6,6,3,64)
    w[2] = zeros(Float32, 1,1,64,1)
    w[3] = winit*randn(Float32, 6,6,64,128)
    w[4] = zeros(Float32, 1,1,128,1)
    w[5] = winit*randn(Float32, 6,6,128,256)
    w[6] = zeros(Float32, 1,1,256,1)
    w[7] = winit*randn(Float32, 6,6,256,512)
    w[8] = zeros(Float32, 1,1,512,1)
    w[9] = winit*randn(Float32, 1, 11*11*512)
    w[10] = zeros(Float32,1,1)
    #return map(a->convert(KnetArray{Float32},a), w)
    return w
  end

  function weights_generator()
    #initalize weights for generator
    winit = 0.02
    w = Array(Any,10)
    w[1] = winit*randn(Float32, 11*11*512,100)
    w[2] = zeros(Float32, 11*11*512,1)
    w[3] = winit*randn(Float32, 6,6,256,512)
    w[4] = zeros(Float32, 1,1,256,1)
    w[5] = winit*randn(Float32, 6,6,128,256)
    w[6] = zeros(Float32, 1,1,128,1)
    w[7] = winit*randn(Float32, 6,6,64,128)
    w[8] = zeros(Float32, 1,1,64,1)
    w[9] = winit*randn(Float32, 6,6,3,64)
    w[10] = zeros(Float32, 1,1,3,1)
    #return map(a->convert(KnetArray{Float32},a), w)
    return w
  end

  function loss_discriminator(w_disc, real_image, fake_image)
  #use for update discriminator's weights
  epsilon=1e-10
  #epsilon=0
  ypred_real = discriminator(w_disc,real_image)
  ynorm_real = log(ypred_real+epsilon)
  J_real = -sum(ynorm_real) / size(ynorm_real,2)
  ypred_fake = discriminator(w_disc,fake_image)
  ynorm_fake = log(1-ypred_fake+epsilon)
  J_fake = -sum(ynorm_fake) / size(ynorm_fake,2)
  return 0.5*J_real + 0.5*J_fake
  end

  function loss_generator(w_gen, w_disc, noise)
  #use for update generator's weights
  epsilon=1e-10
  #epsilon=0
  fake_image = generator(w_gen, noise; mode=0)
  ypred = discriminator(w_disc,fake_image)
  ynorm_fake = log(ypred+epsilon)
  J = -sum(ynorm_fake) / size(ypred,2)
  return J
  end

  grad_gen = grad(loss_generator)
  grad_disc = grad(loss_discriminator)

  function accuracy(w_gen, w_disc, xtst, epoch)
      ncorrect1 = 0
      ncorrect2 = 0
      ninstance = 0
      ninstance2 = 0
      expected_sum1 = 0
      expected_sum2 = 0
      expected_sum = 0
      for i=1:1
      ypred = discriminator(w_disc, xtst[i])
      expected_sum1 += sum(ypred)
      ncorrect1 += sum(ypred[1:1,:] .> 0.5)
      ninstance += size(ypred,2)
      noise = randn(Float32,100,128)
      #noise = convert(KnetArray{Float32}, noise)

      fake_image2 = generator(w_gen, noise;mode=1)
      fake_image = generator(w_gen, noise;mode=0)

      filename = "test_faces_epoch$(epoch)_$(i).png"
      fake_image2 = reshape(fake_image2[:,:,:,1], 176, 176, 3)
      fake_image2 = colorview(RGB, permutedims(fake_image2,[3 2 1]))
      save(filename, (fake_image2))

      ypred2 = discriminator(w_disc,fake_image)
      expected_sum2 += sum(ypred2)
      ncorrect2 += sum((ypred2[1:1,:]) .< 0.5)
      ninstance2 += size(ypred2,2)
      end
      return (ncorrect1/ninstance), (ncorrect2/ninstance2), (expected_sum1+expected_sum2)/(ninstance+ninstance2), (expected_sum1)/(ninstance), (expected_sum2)/(ninstance2)
  end

  function batchnorm(x;epsilon=1e-5)
            mu, sigma = nothing, nothing
            d = ndims(x) == 4 ? (1,2,4) : (2,)
            s = prod(size(x)[[d...]])
            mu = sum(x,d) / s
            sigma = sqrt(epsilon + (sum((x.-mu).^2, d)) / s)

      xhat = (x.-mu) ./ sigma
      return xhat
  end

  function train(w_gen, w_disc, xtrn, epoch, xtst, opts_disc, opts_gen)
    loss_gen = Any[]
    loss_disc = Any[]
    fake_image = Any[]

    noise = randn(Float32,100,128)
    #noise = convert(KnetArray{Float32}, noise)
    fake_image = generator(w_gen, noise; mode=0)
    gradient_gen = grad_gen(w_gen, w_disc, noise)
    gradient_disc = grad_disc(w_disc, xtrn, fake_image)


    for i in 1:length(w_disc)
      update!(w_disc, gradient_disc, opts_disc)
    end

    for i in 1:length(w_gen)
      update!(w_gen, gradient_gen, opts_gen)
    end

  loss_gen = loss_generator(w_gen, w_disc, noise)
  loss_disc = loss_discriminator(w_disc, xtrn, fake_image)
  report(epoch, loss_gen, loss_disc, accuracy(w_gen, w_disc, xtst, epoch))

  return w_gen, w_disc
  end

  main()
