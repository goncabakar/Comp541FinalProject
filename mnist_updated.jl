
for p in ("Knet", "Compat", "GZip", "Images")
  Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet, Compat, GZip, Images

function main()
  batchsize = 128
  info("Loading MNIST...")
  xtrn, xtst0 = load_data()
  xtrn_mb = minibatch4(xtrn, batchsize)
  #xtrn_mb = convert(KnetArray{Float32}, xtrn_mb)
  xtst = minibatch4(xtst0, batchsize)
  #xtst = convert(KnetArray{Float32}, xtst)
  w_gen = weights_generator()
  w_disc = weights_discriminator()
  opts_disc = map(x->Adam(lr=0.0002, beta1=0.5), w_disc)
  opts_gen = map(x->Adam(lr=0.0002, beta1=0.5), w_gen)
  
  noise = randn(Float32,100,batchsize)
  #noise = convert(KnetArray{Float32}, noise)
  fake_image = generator(w_gen, noise,mode=0)
  loss_gen = loss_generator(w_gen, w_disc, noise)
  loss_disc = loss_discriminator(w_disc, xtrn_mb[1], fake_image)
  
  report(0,loss_gen,loss_disc,accuracy(w_gen, w_disc, xtst,0))
  
  @time for epoch=1:20
    train(w_gen, w_disc, xtrn_mb, epoch, xtst, opts_disc, opts_gen)
  end
  
end

report(epoch, lossgen, lossdisc, accuracy)=println((:epoch,epoch,:generatorloss,lossgen,:discriminatorloss,lossdisc,:accuracy_real_fake,accuracy))

function load_data()
  a = gzload("train-images-idx3-ubyte.gz")[17:end]
  b = gzload("t10k-images-idx3-ubyte.gz")[17:end]
  xtrn = convert(Array{Float32}, reshape(2*((a ./ 255)-0.5), 28*28, div(length(a), 784)))
  xtst = convert(Array{Float32}, reshape(2*((b ./ 255)-0.5), 28*28, div(length(b), 784)))
  
  return xtrn, xtst
end

function gzload(file; path="$file", url="http://yann.lecun.com/exdb/mnist/$file")
  isfile(path) || download(url, path)
  f = gzopen(path)
  a = @compat read(f)
  close(f)
  return(a)
end

function minibatch(X, bs)
  
  data = Any[]
  x_mb = Any[]
  for i=1:round(Int,size(X,2)/bs)-1
    x_mb=X[1:784,((i-1)*bs)+1 : i*bs]
    push!(data, x_mb)
  end
  
  return data
end

function minibatch4(x, batchsize)
  data = minibatch(x, batchsize)
  for i=1:length(data)
    data[i] = (reshape(data[i], (28,28,1,batchsize)))
  end
  return data
end

function generator(w, noise; mode=0)
  #use deconvolution layers to generate image from noise
  #use deconv4 for deconvolution operator
  bs = size(noise,2);
  x = w[1]*noise .+ w[2];
  x = reshape(x, 4, 4, 256, bs);
  x = batchnorm(x);
  x = relu(x);
  x = batchnorm(deconv4(w[3],x;padding=2, stride=2) .+ w[4]);
  x = relu(x);
  x = batchnorm(deconv4(w[5],x;padding=2, stride=2) .+ w[6]);
  x = relu(x);
  x = (deconv4(w[7],x;padding=2, stride=2) .+ w[8]);
  
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
  x = (conv4(w[1],x;padding=2, stride=2) .+ w[2])
  x = leaky_relu(x)
  x = batchnorm(conv4(w[3],x;padding=2, stride=2) .+ w[4])
  x = leaky_relu(x)
  x = batchnorm(conv4(w[5],x;padding=2, stride=2) .+ w[6])
  x = leaky_relu(x)
  x = mat(x)
  x = w[7]*x .+ w[8]
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
  w = Array(Any,8)
  w[1] = winit*randn(Float32, 6,6,1,64)
  w[2] = zeros(Float32, 1,1,64,1)
  w[3] = winit*randn(Float32, 6,6,64,128)
  w[4] = zeros(Float32, 1,1,128,1)
  w[5] = winit*randn(Float32, 5,5,128,256)
  w[6] = zeros(Float32, 1,1,256,1)
  w[7] = winit*randn(Float32, 1, 4*4*256)
  w[8] = zeros(Float32,1,1)
  #return map(a->convert(KnetArray{Float32},a), w)
  return w
end

function weights_generator()
  #initalize weights for generator
  winit = 0.02
  w = Array(Any,8)
  w[1] = winit*randn(Float32, 4*4*256,100)
  w[2] = zeros(Float32, 4*4*256,1)
  w[3] = winit*randn(Float32, 5,5,128,256)
  w[4] = zeros(Float32, 1,1,128,1)
  w[5] = winit*randn(Float32, 6,6,64,128)
  w[6] = zeros(Float32, 1,1,64,1)
  w[7] = winit*randn(Float32, 6,6,1,64)
  w[8] = zeros(Float32, 1,1,1,1)
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
  fake_image = generator(w_gen, noise;mode=0)
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
  for i=1:20
    
    ypred = discriminator(w_disc, xtst[i])
    expected_sum1 += sum(ypred)
    ncorrect1 += sum(ypred[1:1,:] .> 0.5)
    ninstance += size(ypred,2)
    noise = randn(Float32,100,128)
    fake_image2 = generator(w_gen, noise;mode=1)
    fake_image = generator(w_gen, noise;mode=0)
    filename = "test_mnist_epoch$(epoch)_$(i).png"
    fake_image2 = reshape(fake_image2[:,:,:,1], 28, 28)
    fake_image2 = colorview(Gray, fake_image2');
    save(filename, fake_image2)
    
    for w in 1:length(w_gen)
      writedlm(filename*"_weight$w.txt", w_gen[w])
    end
    writedlm(filename*"_noise.txt", noise)
    
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
  
  #train_bs = 100
  train_bs = round(Int,60000/128)-1
  for mb=1:train_bs
    noise = randn(Float32,100,128)
    fake_image = generator(w_gen, noise; mode=0)
    gradient_gen = grad_gen(w_gen, w_disc, noise)
    gradient_disc = grad_disc(w_disc, xtrn[mb], fake_image)
    
    for i in 1:length(w_disc)
      update!(w_disc[i], gradient_disc[i], opts_disc[i])
    end
    for i in 1:length(w_gen)
      update!(w_gen[i], gradient_gen[i], opts_gen[i])
    end
  end
  
  noise = randn(Float32,100,128)
  loss_gen = loss_generator(w_gen, w_disc, noise)
  loss_disc = loss_discriminator(w_disc, xtrn[train_bs], fake_image)
  report(epoch, loss_gen, loss_disc, accuracy(w_gen, w_disc, xtst, epoch))
  
  return w_gen, w_disc
end

main()
