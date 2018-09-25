[toc]

# Git

[Git教程 - 廖雪峰的官方网站](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)

# Virtualbox

1. virtualbox -> 设置 -> 常规 -> 高级 -> 共享粘贴板：双向，拖放：双向。
2. virtualbox -> 设置 -> 显示 -> 不要勾选高清(HiDPI)支持，很卡顿，建议勾选启动3D加速和启用和2D视频加速，有些图形软件可能会用到。
3. 文件共享
    - virtualbox -> 设置 -> 共享文件夹 -> 添加一个文件夹，勾选固定分配和自动挂载 -> 点OK保存设置。
    - 启动windows，Device -> Insert Guest Additions CD Image -> 进入我的电脑 -> 打开Guest Additions CD Image -> 运行VBoxWindowsAdditions-amd64.exe。
    - 重启windows后，共享文件夹会显示在我的电脑的网络位置里。

# Python
## TensorFlow

### variable_scope, name_scope, arg_scope
- variable_scope
```python
__init__(
    name_or_scope,  # string or VariableScope: the scope to open.
    default_name=None,  # The default name to use if the name_or_scope argument is None, this name will be uniquified. If name_or_scope is provided it won't be used and therefore it is not required and can be None.
    values=None,  # The list of Tensor arguments that are passed to the op function.
    initializer=None,  # default initializer for variables within this scope.
    regularizer=None,  # default regularizer for variables within this scope.
    caching_device=None,  # default caching device for variables within this scope.
    partitioner=None,  # default partitioner for variables within this scope.
    custom_getter=None,  # default custom getter for variables within this scope.
    reuse=None,  # True, None, or tf.AUTO_REUSE; if True, we go into reuse mode for this scope as well as all sub-scopes; if tf.AUTO_REUSE, we create variables if they do not exist, and return them otherwise; if None, we inherit the parent scope's reuse flag. When eager execution is enabled, new variables are always created unless an EagerVariableStore or template is currently active.
    dtype=None,  # type of variables created in this scope (defaults to the type in the passed scope, or inherited from parent scope).
    use_resource=None,  # If False, all variables will be regular Variables. If True, experimental ResourceVariables with well-defined semantics will be used instead. Defaults to False (will later change to True). When eager execution is enabled this argument is always forced to be True.
    constraint=None,  # An optional projection function to be applied to the variable after being updated by an Optimizer (e.g. used to implement norm constraints or value constraints for layer weights). The function must take as input the unprojected Tensor representing the value of the variable and return the Tensor for the projected value (which must have the same shape). Constraints are not safe to use when doing asynchronous distributed training.
    auxiliary_name_scope=True  # If True, we create an auxiliary name scope with the scope. If False, we don't create it. Note that the argument is not inherited, and it only takes effect for once when creating. You should only use it for re-entering a premade variable scope.
)

```
- name_scope
```python
__init__(
    name,  # The name argument that is passed to the op function.
    default_name=None,  # The default name to use if the name argument is None.
    values=None  # The list of Tensor arguments that are passed to the op function.
)
```
- 用tf.Variable()的话每次都会新建变量。但是大多数时候我们是希望重用一些变量，所以就用到了tf.get_variable()，它拥有一个变量检查机制，会检测变量是否存在及是否为共享变量，如果变量不存在则创建，如果已经存在的变量没有设置为共享变量，TensorFlow 运行到第二个拥有相同名字的变量的时候，就会报错。
```python 
import tensorflow as tf

with tf.name_scope('name_scope_1'):
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
    var2 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))
    print(var2.name, sess.run(var2))

# ValueError: Variable var1 already exists, disallowed. Did you mean 
# to set reuse=True in VarScope? Originally defined at:
# var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
```
- tf.Variable()会自动检测有没有变量重名，如果这个变量已存在，则后缀会增加0、1、2等数字编号予以区别。
```python
import tensorflow as tf

with tf.name_scope("a_name_scope"):
    var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
    var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(var1.name) # var1:0
    print(sess.run(var1)) # [ 1.]
    print(var2.name) # a_name_scope/var2:0
    print(sess.run(var2)) # [ 2.]
    print(var21.name) # a_name_scope/var2_1:0
    print(sess.run(var21)) # [ 2.0999999]
    print(var22.name) # a_name_scope/var2_2:0
    print(sess.run(var22)) # [ 2.20000005]
```
- 在 tf.name_scope下时，tf.get_variable()创建的变量名不受 name_scope 的影响。
```python
import tensorflow as tf

with tf.variable_scope("foo"):
    with tf.name_scope("bar"):
        v = tf.get_variable("v", [1])
        x = 1.0 + v
assert v.name == "foo/v:0"  # True
assert x.op.name == "foo/bar/add"  # True
```
- 要共享变量，需要使用tf.variable_scope()。
```python
import tensorflow as tf

with tf.variable_scope('variable_scope_y') as scope:
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
    scope.reuse_variables()  # 设置共享变量
    # or tf.get_variable_scope().reuse_variables()
    var1_reuse = tf.get_variable(name='var1')
    var2 = tf.Variable(initial_value=[2.], name='var2', dtype=tf.float32)
    var2_reuse = tf.Variable(initial_value=[2.], name='var2', dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))
    print(var1_reuse.name, sess.run(var1_reuse))
    print(var2.name, sess.run(var2))
    print(var2_reuse.name, sess.run(var2_reuse))
# 输出结果：
# variable_scope_y/var1:0 [-1.59682846]
# variable_scope_y/var1:0 [-1.59682846]   可以看到变量var1_reuse重复使用了var1
# variable_scope_y/var2:0 [ 2.]
# variable_scope_y/var2_1:0 [ 2.]
```
也可以这样
```python
with tf.variable_scope('foo') as foo_scope:
    v = tf.get_variable('v', [1])
with tf.variable_scope('foo', reuse=True):
    v1 = tf.get_variable('v')
assert v1 == v
```
或者这样：
```python
with tf.variable_scope('foo') as foo_scope:
    v = tf.get_variable('v', [1])
with tf.variable_scope(foo_scope, reuse=True):
    v1 = tf.get_variable('v')
assert v1 == v
```
- arg_scope
```python
tf.contrib.framework.arg_scope(
    list_ops_or_scope,
    **kwargs
)

# list_ops_or_scope: List or tuple of operations to set argument scope for or a dictionary containing the current scope. When list_ops_or_scope is a dict, kwargs must be empty. When list_ops_or_scope is a list or tuple, then every op in it need to be decorated with @add_arg_scope to work.
# **kwargs: keyword=value that will define the defaults for each op in list_ops. All the ops need to accept the given set of arguments.
```
arg_scope is a way to avoid repeating providing the same arguments over and over again to the same layer types.
```python
# Example of how to use tf.contrib.framework.arg_scope.
from third_party.tensorflow.contrib.layers.python import layers
  arg_scope = tf.contrib.framework.arg_scope
  with arg_scope([layers.conv2d], padding='SAME',
                 initializer=layers.variance_scaling_initializer(),
                 regularizer=layers.l2_regularizer(0.05)):
    net = layers.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
    net = layers.conv2d(net, 256, [5, 5], scope='conv2')

# The first call to conv2d will behave as follows.
layers.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
              initializer=layers.variance_scaling_initializer(),
              regularizer=layers.l2_regularizer(0.05), scope='conv1')

# The second call to conv2d will also use the arg_scope's default for padding.
layers.conv2d(inputs, 256, [5, 5], padding='SAME',
              initializer=layers.variance_scaling_initializer(),
              regularizer=layers.l2_regularizer(0.05), scope='conv2')

# Example of how to reuse an arg_scope.
with arg_scope([layers.conv2d], padding='SAME',
                 initializer=layers.variance_scaling_initializer(),
                 regularizer=layers.l2_regularizer(0.05)) as sc:
    net = layers.conv2d(net, 256, [5, 5], scope='conv1')

with arg_scope(sc):
    net = layers.conv2d(net, 256, [5, 5], scope='conv2')
```

### Feature columns and transformations

A FeatureColumn represents a single feature in your data. A FeatureColumn may represent a quantity like 'height', or it may represent a category like 'eye_color' where the value is drawn from a set of discrete possibilities like {'blue', 'brown', 'green'}.

In the case of both continuous features like 'height' and categorical
features like 'eye_color', a single value in the data might get transformed into a sequence of numbers before it is input into the model. The FeatureColumn abstraction lets you manipulate the feature as a single semantic unit in spite of this fact. You can specify transformations and select features to include without dealing with specific indices in the tensors you feed into the model.

- Sparse columns
```pythonpython
eye_color = tf.contrib.layers.sparse_column_with_keys(column_name="eye_color", keys=["blue", "brown", "green"])
```
```python
# When you don't know all possible values.
education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
```
- Feature crosses
```python
sport = tf.contrib.layers.sparse_column_with_hash_bucket("sport", hash_bucket_size=1000)
city = tf.contrib.layers.sparse_column_with_hash_bucket("city", hash_bucket_size=1000)
sport_x_city = tf.contrib.layers.crossed_column([sport, city], hash_bucket_size=int(1e4))
```
- Continuous columns
```python
age = tf.contrib.layers.real_valued_column("age")
```
- Bucketization
```python
# Bucketization divides the range of possible values into subranges called buckets.
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```
- Input function
    FeatureColumns provide a specification for the input data for your model, indicating how to represent and transform the data. But they ``do not`` provide the data itself. You provide the data through an input function, e.g., tf.contrib.layers.input_from_feature_columns.

    The input function must return a dictionary of tensors. Each key corresponds to the name of a FeatureColumn. Each key's value is a tensor containing the values of that feature for all data instances.

```python
input_from_feature_column(
    columns_to_tensors,
    feature_columns,
    weight_collections=None,
    trainable=True,
    scope=None,
    cols_to_outs=None
)

# columns_to_tensors: A mapping from feature column to tensors. 'string' key means a base feature (not-transformed). It can have FeatureColumn as a key too. That means that FeatureColumn is already transformed by input pipeline.
# feature_columns: A set containing all the feature columns. All items in the set should be instances of classes derived by FeatureColumn.
# weight_collections: List of graph collections to which weights are added.
# trainable: If True also add variables to the graph collection GraphKeys.TRAINABLE_VARIABLES (see tf.Variable).
# scope: Optional scope for variable_scope.
# cols_to_outs: Optional dict from feature column to output tensor, which is concatenated into the returned tensor.
```


- My understanding: the columns_to_tensors contains the raw data (e.g., for gender, raw data may be string tensors), the input_from_feature_column transforms the raw data to features based on the definition of feature_columns (e.g., to indicator features, [0, 1] for "female" and [1, 0] for "male").
