# Lessons Learned From the TensorFlow Workshop

- Install TF per pip package in a virtualenv
- _always_ be aware of **TensorFlow mechanics**
  - In few words: Using the python API we define the computation _graph_ first, outside a session. The python variables assigned while doing so do not trigger computations directly, but rather represent variables/constants (edges) and operations (nodes). In order to evaluate, "fetch", values from the graph, we open a `tf.Session()` and either execute `sess.run(tensor)` or `tensor.eval()`. TensorFlow will only then place the computations (of the graph) necessary for evaluating the the fetched tensor on the computing device, e.g. GPU. This is a _very_ brief summary; you can read more [here](https://www.tensorflow.org/programmers_guide/)
  - Q: Is a statement supposed to reside inside/outside Session?
  - Q: What does variable resemble? Operation or tensor?
  - If you put `a = tf.constant(3)` in loop, even though within the loops it is assigned to the same python variable, you're creating infinite constants of one kind! TensorFlow will create infinite unique names. The same can happen for operations, e.g. assume you have an `iterator` to a Dataset; Using `sess.run(iterator.get_next())` instead of defining the operation `next_elem_op = iterator.get_next()` _once_ and calling it: `sess.run(next_elem_op)` would result in infinite input pipelines being instantiated, filling the memory.
- Use the [`tf.data.Dataset` API](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), it makes everything easier.
- Parsing, preprocessing & augmentation steps are done on CPU. Reserve GPU power for parallelizable operations, e.g. matrix multiplications, convolutional layers, backpropagation. TensorFlow does this automatically.
- tf.constant or tf.Variable? Set attribute trainable correctly.
- Try to avoid multiple `sess.run()` calls. Each time you evaluate a Tensor like this you get the values as numpy array in plain Python, exiting the optimized TensorFlow graph computations in CUDA/optimized C.
- RGB vs BGR ;-) TensorFlow reads images as RGB per default, OpenCV assumes BGR.
- Don't write image summaries, they just bloat and clutter the tfevents files. Write images to disk.
- Don't be a hero! (when it comes to DNN architectures). Choose easy to use implementations over cutting edge architectures from dubious sources.
- If you have a very common task at hand, a TensorFlow [Estimator](https://www.tensorflow.org/programmers_guide/estimators) could already suffice.

