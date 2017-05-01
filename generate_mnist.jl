for p in ("Knet", "Compat", "GZip", "Images")
  Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet, Compat, GZip, Images

function main()

  w_gen = weights_generator(20, 20)
  w_gen = map(a->convert(Array{Float32},a), w_gen)

  epoch = 17
  i = 7
  filename = "test_mnist_epoch$(epoch)_$(i).png"
  noise1 = readdlm(filename*"_noise.txt")
  noise1 = convert(Array{Float32}, noise1[:, 1:1])

  epoch = 20
  i = 9
  filename = "test_mnist_epoch$(epoch)_$(i).png"
  noise2 = readdlm(filename*"_noise.txt")
  noise2 = convert(Array{Float32}, noise2[:, 1:1])

  noise = Array(Any,11)
  noise[1] = noise1
  noise[11] = noise2
  diff = (noise2 - noise1)/10

  filename2 = "generated0.png"
  fake_image = generator(w_gen, noise[1],mode=1)
  fake_image = reshape(fake_image[:,:,:,1], 28, 28)
  fake_image = colorview(Gray, fake_image');
  save(filename2, fake_image)

  for k = 1:9
    noise[k+1] = noise[k] + diff
    filename2 = "generated$k.png"
    fake_image = generator(w_gen, noise[k+1],mode=1)
    fake_image = reshape(fake_image[:,:,:,1], 28, 28)
    fake_image = colorview(Gray, fake_image');
    save(filename2, fake_image)
  end
  filename2 = "generated10.png"
  fake_image = generator(w_gen, noise[11],mode=1)
  fake_image = reshape(fake_image[:,:,:,1], 28, 28)
  fake_image = colorview(Gray, fake_image');
  save(filename2, fake_image)


  #for epoch = 1:20
  #i = 20
  #filename2 = "generated$epoch.png"
  #w_gen = weights_generator(epoch, i)
  #w_gen = map(a->convert(Array{Float32},a), w_gen)
  #fake_image = generator(w_gen, noise,mode=1)
  #fake_image = reshape(fake_image[:,:,:,1], 28, 28)
  #fake_image = colorview(Gray, fake_image');
  #save(filename2, fake_image)
  #println("Image is saved.")
  #end


end


function generator(w, noise; mode=1)
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
  #x = batchnorm(deconv4(w[7],x;padding=2, stride=2) .+ w[8]);
  x = (deconv4(w[7],x;padding=2, stride=2) .+ w[8]);

  if mode == 1
    fake_images = sigm(x);
  else
    fake_images = tanh(x);
  end
  println("Image is generated.")
  return fake_images
end


function weights_generator(epoch, i)
  #initalize weights for generator


  w = Array(Any,8)

  println("Weights are initiliazed.")
  filename = "test_mnist_epoch$(epoch)_$(i).png"
  for k = 1:8
  w[k] = readdlm(filename*"_weight$k.txt")
  end

  w[1] = reshape(w[1], 4*4*256,100)
  w[2] = reshape(w[2], 4*4*256,1)
  w[3] = reshape(w[3], 5,5,128,256)
  w[4] = reshape(w[4], 1,1,128,1)
  w[5] = reshape(w[5], 6,6,64,128)
  w[6] = reshape(w[6], 1,1,64,1)
  w[7] = reshape(w[7], 6,6,1,64)
  w[8] = reshape(w[8], 1,1,1,1)
  #return map(a->convert(KnetArray{Float32},a), w)
  return w
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

main()
