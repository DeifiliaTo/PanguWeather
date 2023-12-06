from networks.pangu import PanguModel

class Train():
  
  # TODO: input parameters are based off of FCN
  # Adapt to make sense for our case.

  def __init__(self, params = {}, world_rank=0):
    self.params = params
    self.world_rank = world_rank
    self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# TODO: logging information
#    logging.info('rank %d, begin data loader init'%world_rank)
    self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(params, params.train_data_path, dist.is_initialized(), train=True)
    self.valid_data_loader, self.valid_dataset = get_data_loader(params, params.valid_data_path, dist.is_initialized(), train=False)
    self.loss_obj = LpLoss()
#    logging.info('rank %d, data loader initialized'%world_rank)

  '''Training code'''
  # Initialize the model, for some APIs some adaptation is needed to fit hardwares
  model = PanguModel()

  # Train single Pangu-Weather model
  epochs = 100
  for i in arange(epochs):
    # For each epoch, we iterate from 1979 to 2017
    # dataset_length is the length of your training data, e.g., the sample between 1979 and 2017
    for step in arange(2):
      # Load weather data at time t as the input; load weather data at time t+1/3/6/24 as the output
      # Note the data need to be randomly shuffled
      # Note the input and target need to be normalized, see Inference() for details
      input, input_surface, target, target_surface = LoadData(step)

      # Call the model and get the output
      output, output_surface = model(input, input_surface)

      # We use the MAE loss to train the model
      # The weight of surface loss is 0.25
      # Different weight can be applied for differen fields if needed
      loss = TensorAbs(output-target) + TensorAbs(output_surface-target_surface) * 0.25

      # Call the backward algorithm and calculate the gratitude of parameters
      Backward(loss)

      # Update model parameters with Adam optimizer
      # The learning rate is 5e-4 as in the paper, while the weight decay is 3e-6
      # A example solution is using torch.optim.adam
      UpdateModelParametersWithAdam()

  # Save the model at the end of the training stage
  SaveModel()