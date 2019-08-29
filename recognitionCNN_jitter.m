
%% Create NN

trainOpts.batchSize = 30 ;
trainOpts.numEpochs = 100 ;
trainOpts.continue = true ;
% trainOpts.useGpu = false ;
trainOpts.learningRate = 0.001 ;
trainOpts.numEpochs = 500 ;
trainOpts.expDir = 'data1/chars-experiment' ;

% Initialize
net = initializeCharacterCNN_jitter() ;

% Take the average image out
imageMean = mean(imdb.images.data(:)) ;
imdb.images.data = imdb.images.data - imageMean ;

% Call training function in MatConvNet
[net,info] = cnn_train(net, imdb, @getBatch, trainOpts) ;

% Save the result for later use
net.layers(end) = [] ;
net.imageMean = imageMean ;
save('data1/chars-experiment/charscnn.mat', '-struct', 'net') ;


% -------------------------------------------------------------------------
%  apply the model
% -------------------------------------------------------------------------

% Load the CNN learned before
net = load('data1/chars-experiment/charscnn.mat') ;
%net = load('data/chars-experiment/charscnn-jit.mat') ;

correct=0;
total=0;
for i=1:26*20
    if imdb.images.set(i)==2
      res = vl_simplenn(net, imdb.images.data(:,:,i)) ;
      [~,guess]=max(res(8).x);
      total=total+1;
      if guess==imdb.images.label(i)
          correct=correct+1;
      end
    end
end

% Display accuracy
disp('Accuracy:')
disp(correct/total)