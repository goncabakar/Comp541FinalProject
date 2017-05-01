for p in ("Knet", "Compat", "GZip", "Images")
  Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet, Compat, GZip, Images

function main()

  w_gen = weights_generator(1, 1)
  w_gen = map(a->convert(Array{Float32},a), w_gen)

  noise = 2*(rand(Float32,100,128).-0.5)
  fake_image = generator(w_gen, noise,mode=0)


  for i = 1:100
  filename2 = "generated2_$i.png"
  fake_image1 = reshape(fake_image[:,:,:,i], 32, 32, 3)
  fake_image2 = colorview(RGB, permutedims(fake_image1,[3 2 1]))
  save(filename2, fake_image2)
  println("Image is saved.")
  end


end


function generator(w, noise; mode=0)
  #use deconvolution layers to generate image from noise
  #use deconv4 for deconvolution operator
  bs = size(noise,2);
  x = w[1]*noise .+ w[2];
  x = reshape(x, 2, 2, 1024, bs);
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
    fake_images = (tanh(x).+1)/2;
  end
  return fake_images
end


function weights_generator(epoch, i)
  #initalize weights for generator


  w = Array(Any,10)

  println("Weights are initiliazed.")
  filename = "test_faces_epoch$(epoch)_$(i).png"
  for k = 1:10
  w[k] = readdlm(filename*"_weight$k.txt")
  end

  w[1] = reshape(w[1], 2*2*1024,100)
  w[2] = reshape(w[2], 2*2*1024,1)
  w[3] = reshape(w[3], 6,6,512,1024)
  w[4] = reshape(w[4], 1,1,512,1)
  w[5] = reshape(w[5], 6,6,256,512)
  w[6] = reshape(w[6], 1,1,256,1)
  w[7] = reshape(w[7], 6,6,128,256)
  w[8] = reshape(w[8], 1,1,128,1)
  w[9] = reshape(w[9], 6,6,3,128)
  w[10] = reshape(w[10], 1,1,3,1)

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
