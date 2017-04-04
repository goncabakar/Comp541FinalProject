
for p in ("Knet", "Compat", "GZip")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet, Compat, GZip

function main()
  batchsize = 100
  info("Loading MNIST...")
  xtrn, xtst0 = load_data()
  xtrn_mb = minibatch4(xtrn, batchsize)
  xtst = minibatch4(xtst0, batchsize)
  w_gen = weights_generator()
  w_disc = weights_discriminator()

  report(0,0,0,accuracy(w_gen, w_disc, xtst))

  @time for epoch=1:5
        train(w_gen, w_disc, xtrn_mb, epoch, xtst)
        #report(epoch)
  end

end

report(epoch, lossgen, lossdisc, accuracy)=println((:epoch,epoch,:generatorloss,lossgen,:discriminatorloss,lossdisc,:accuracy,accuracy))

function load_data()
    a = gzload("train-images-idx3-ubyte.gz")[17:end]
    b = gzload("t10k-images-idx3-ubyte.gz")[17:end]
    xtrn = convert(Array{Float32}, reshape(a ./ 255, 28*28, div(length(a), 784)))
    xtst = convert(Array{Float32}, reshape(b ./ 255, 28*28, div(length(b), 784)))

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
    for i=1:round(Int,size(X,2)/bs)
    x_mb=X[1:784,((i-1)*bs)+1 : i*bs]
    push!(data, x_mb)
    end

    return data
end

function minibatch4(x, batchsize)
    data = minibatch(x, batchsize)
    for i=1:length(data)
        #x = data[i]
        data[i] = (reshape(data[i], (28,28,1,batchsize)))
    end
    return data
end
function generator(w_gen, noise)
  #use deconvolution layers to generate image from noise
  #use deconv4 for deconvolution operator
  fake_images = rand(Float32,28,28,1,100)
  return fake_images
end

function discriminator(w, x)
  #decides whether image is fake or real
  #inputs can be generated images or real images from dataset
  x = leakly_relu(conv4(w[1],x;padding=2, stride=2) .+ w[2])
  x = batchnorm(x)
  x = leakly_relu(conv4(w[3],x;padding=2, stride=2) .+ w[4])
  x = batchnorm(x)
  x = leakly_relu(conv4(w[5],x;padding=2, stride=2) .+ w[6])
  x = batchnorm(x)
  x = mat(x)
  x = w[7]*x .+ w[8]
  return x
end

function leakly_relu(x)
return  max(0.01*x,x)
end

function weights_discriminator()
  #initalize weights for discriminator
  #it is a binary classifier
  winit = 0.1
  w = Array(Any,8)
  w[1] = winit*randn(Float32, 5,5,1,64)
  w[2] = zeros(Float32, 1,1,64,1)
  w[3] = winit*randn(Float32, 5,5,64,128)
  w[4] = zeros(Float32, 1,1,128,1)
  w[5] = winit*randn(Float32, 5,5,128,256)
  w[6] = zeros(Float32, 1,1,256,1)
  w[7] = winit*randn(Float32, 2, 4*4*256)
  w[8] = zeros(Float32,2,1)
  return w
end

function weights_generator()
  #initalize weights for generator
  num_filter = 10
  filter_size = 3
  winit = 0.1
  w = Array(Any,4)
  w[1] = winit*randn(Float32, filter_size,filter_size,1,num_filter)
  w[2] = zeros(Float32, 1,1,num_filter,1)
  w[3] = winit*randn(Float32, 10, 13*13*num_filter)
  w[4] = zeros(Float32,10,1)
  return w
end

function loss_discriminator(w_disc, w_gen, real_image, noise)
#use for update discriminator's weights
ypred_real = discriminator(w_disc,real_image)
ynorm_real = -logp(ypred_real,2)
#ygold_true = ones(Float32,1,100)
J_real = sum(ynorm_real[1,:]) / size(ynorm_real,2)
fake_image = generator(w_gen, noise)
ypred_fake = discriminator(w_disc,fake_image)
ynorm_fake = -logp(ones(Float32,2,100) - ypred_fake,2)
#ygold_false = ones(Float32,1,100)
J_fake = sum(ynorm_fake[2,:]) / size(ynorm_fake,2)
return J_real + J_fake
#return J_fake
end

function loss_generator(w_gen, w_disc, noise)
#use for update generator's weights
fake_image = generator(w_gen, noise)
ypred = discriminator(w_disc,fake_image)
ynorm_fake = -logp(ypred,1)
J = sum(ynorm_fake[2,:]) / size(ypred,2)
return J
end

grad_gen = grad(loss_generator)
grad_disc = grad(loss_discriminator)

function accuracy(w_gen, w_disc, xtst)
    ncorrect1 = 0
    ncorrect2 = 0
    ninstance = 0
    for i=1:100
    ypred = discriminator(w_disc, xtst[i])
    ypred = -logp(ypred, 1)
    #ncorrect += sum(0.5 .> (ypred))
    ncorrect1 += sum((ypred[1,:] .== maximum(ypred,1)))
    ninstance += size(ypred,2)

    noise = rand(Float32,100,1)
    fake_image = generator(w_gen, noise)
    ypred2 = discriminator(w_disc,fake_image)
    ynorm_fake = -logp(ypred2,1)
    ncorrect2 += sum((ypred[2,:] .== maximum(ypred,1)))
    end
    return (ncorrect1/ninstance), (ncorrect2/ninstance)
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

function train(w_gen, w_disc, xtrn, epoch, xtst)
  loss_gen = Any[]
  loss_disc = Any[]
  noise = rand(Float32,100,1)
  train_bs = 10
  for mb=1:train_bs
  #for mb=1:600
#gradient_gen = grad_gen(w_gen, w_disc, noise)
gradient_disc = grad_disc(w_disc, w_gen, xtrn[mb], noise)

lr = 0.005
for i in 1:length(w_gen)
  #opts_gen = Adam()
  w_gen[i] = w_gen[i] - lr * 0
  #update!(w_gen, gradient_gen, opts_gen)
end
opts_disc = map(x->Adam(), w_disc)
for i in 1:length(w_disc)
    #w_disc[i] = w_disc[i] - lr * gradient_disc[i]
    update!(w_disc, gradient_disc, opts_disc)
end
end

loss_gen = loss_generator(w_gen, w_disc, noise)
loss_disc = loss_discriminator(w_disc, w_gen, xtrn[train_bs], noise)
report(epoch, loss_gen, loss_disc, accuracy(w_gen, w_disc, xtst))

return w_gen, w_disc
end

main()
