from keras.callbacks import TensorBoard
from tensorflow.summary import create_file_writer, scalar


class CustomTensorBoard(TensorBoard):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.writer = create_file_writer(self.log_dir)

	def set_model(self, model):
		pass

	def log(self, step, **stats):
		# self._write_logs(stats, step)
		with self.writer.as_default():
			for stat in stats:
				scalar(stat, stats[stat], step=step)