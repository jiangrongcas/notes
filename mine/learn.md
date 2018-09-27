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


## 下划线含义

- 单前导下划线：_var
- 单末尾下划线：var_
- 双前导下划线：__var
- 双前导和末尾下划线：\_\_var__
- 单下划线：_

- #### 单前导下划线 _var
    当涉及到变量和方法名称时，单个下划线前缀有一个约定俗成的含义。它是对程序员的一个提示 - 意味着Python社区一致认为它应该是什么意思，但程序的行为不受影响。

    看看下面的例子：
    ```python
    class Test:
        def __init__(self):
            self.foo = 11
            self._bar = 23
    ```
    ```
    >>> t = Test()
    >>> t.foo
    11
    >>> t._bar
    23
    ```
    **但是**，前导下划线的确会影响从模块中导入名称的方式，看例子：
    ```python
    # This is my_module.py:
    def external_func():
        return 23

    def _internal_func():
        return 42
    ```
    现在，如果使用通配符从模块中导入所有名称，则Python不会导入带有前导下划线的名称（除非模块定义了覆盖此行为的__all__列表）：
    ```
    >>> from my_module import *
    >>> external_func()
    23
    >>> _internal_func()
    NameError: "name '_internal_func' is not defined"
    ```
    顺便说一下，应该避免通配符导入，因为它们使名称空间中存在哪些名称不清楚。 为了清楚起见，坚持常规导入更好。与通配符导入不同，常规导入不受前导单个下划线命名约定的影响：
    ```
    >>> import my_module
    >>> my_module.external_func()
    23
    >>> my_module._internal_func()
    42
    ```
- #### 单末尾下划线 var_
    有时候，一个变量的最合适的名称已经被一个关键字所占用。 因此，像class或def这样的名称不能用作Python中的变量名称。 在这种情况下，你可以附加一个下划线来解决命名冲突：
    ```
    >>> def make_object(name, class):
    SyntaxError: "invalid syntax"
    ```
    可改为def make_object(name, class_)。
- #### 双前导下划线 __var
    到目前为止，我们所涉及的所有命名模式的含义，来自于已达成共识的约定。 而对于以双下划线开头的Python类的属性（包括变量和方法），情况就有点不同了。
    
    双下划线前缀会导致Python解释器重写属性名称，以避免子类中的命名冲突。
    
    这也叫做名称修饰（name mangling） - 解释器更改变量的名称，以便在类被扩展的时候不容易产生冲突.

    ```python
    class Test:
        def __init__(self):
            self.foo = 11
            self._bar = 23
            self.__baz = 23
    ```
    用内置的dir()函数来看看这个对象的属性：
    ```
    >>> t = Test()
    >>> dir(t)
    ['_Test__baz', '__class__', '__delattr__', '__dict__', '__dir__',
    '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__',
    '__gt__', '__hash__', '__init__', '__le__', '__lt__', '__module__',
    '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__',
    '__setattr__', '__sizeof__', '__str__', '__subclasshook__',
    '__weakref__', '_bar', 'foo']
    ```
    以上是这个对象属性的列表：
    - self.foo变量在属性列表中显示为未修改为foo。
    - self._bar的行为方式相同 - 它以_bar的形式显示在类上。 就像我之前说过的，在这种情况下，前导下划线仅仅是一个约定。 给程序员一个提示而已。
    - 然而，对于self.__baz而言，情况看起来有点不同。 当你在该列表中搜索__baz时，你会看不到有这个名字的变量。
  
    __baz出什么情况了？

    如果你仔细观察，你会看到此对象上有一个名为_Test__baz的属性。 这就是Python解释器所做的名称修饰。 它这样做是为了防止变量在子类中被重写。

    让我们创建另一个扩展Test类的类，并尝试重写构造函数中添加的现有属性：
    ```python
    class ExtendedTest(Test):
        def __init__(self):
            super().__init__()
            self.foo = 'overridden'
            self._bar = 'overridden'
            self.__baz = 'overridden'
    ```
    现在，你认为foo，_bar和__baz的值会出现在这个ExtendedTest类的实例上吗？ 我们来看一看：
    ```
    >>> t2 = ExtendedTest()
    >>> t2.foo
    'overridden'
    >>> t2._bar
    'overridden'
    >>> t2.__baz
    AttributeError: "'ExtendedTest' object has no attribute '__baz'"
    ```
    当我们尝试查看t2 .__ baz的值时，为什么我们会得到AttributeError？ 名称修饰被再次触发了！ 事实证明，这个对象甚至没有__baz属性：
    ```
    >>> dir(t2)
    ['_ExtendedTest__baz', '_Test__baz', '__class__', '__delattr__',
    '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
    '__getattribute__', '__gt__', '__hash__', '__init__', '__le__',
    '__lt__', '__module__', '__ne__', '__new__', '__reduce__',
    '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__',
    '__subclasshook__', '__weakref__', '_bar', 'foo', 'get_vars']
    ```
    正如你可以看到__baz变成_ExtendedTest__baz以防止意外修改：
    ```
    >>> t2._ExtendedTest__baz
    'overridden'
    ```
    但原来的_Test__baz还在：
    ```
    >>> t2._Test__baz
    42
    ```
    双下划线名称修饰对程序员是完全透明的。 下面的例子证实了这一点：
    ```python
    class ManglingTest:
        def __init__(self):
            self.__mangled = 'hello'

        def get_mangled(self):
            return self.__mangled
    ```
    ```
    >>> ManglingTest().get_mangled()
    'hello'
    >>> ManglingTest().__mangled
    AttributeError: "'ManglingTest' object has no attribute '__mangled'"
    ```
    名称修饰是否也适用于方法名称？ 是的，也适用。名称修饰会影响在一个类的上下文中，以两个下划线字符（"dunders"）开头的所有名称：
    ```python
    class MangledMethod:
        def __method(self):
            return 42

        def call_it(self):
            return self.__method()
    ```
    ```
    >>> MangledMethod().__method()
    AttributeError: "'MangledMethod' object has no attribute '__method'"
    >>> MangledMethod().call_it()
    42
    ```
    这是另一个也许令人惊讶的运用名称修饰的例子：
    ```python
    _MangledGlobal__mangled = 23

    class MangledGlobal:
        def test(self):
            return __mangled
    ```
    ```
    >>> MangledGlobal().test()
    23
    ```
    在这个例子中，我声明了一个名为_MangledGlobal__mangled的全局变量。然后我在名为MangledGlobal的类的上下文中访问变量。由于名称修饰，我能够在类的test()方法内，以__mangled来引用_MangledGlobal__mangled全局变量。Python解释器自动将名称__mangled扩展为_MangledGlobal__mangled，因为它以两个下划线字符开头。这表明名称修饰不是专门与类属性关联的。它适用于在类上下文中使用的两个下划线字符开头的任何名称。

- #### 双前导和双末尾下划线 \_\_var__
    也许令人惊讶的是，如果一个名字同时以双下划线开始和结束，则不会应用名称修饰。 由双下划线前缀和后缀包围的变量不会被Python解释器修改：
    ```python
    class PrefixPostfixTest:
        def __init__(self):
        self.__bam__ = 42
    ```
    ```
    >>> PrefixPostfixTest().__bam__
    42
    ```
    但是，Python保留了有双前导和双末尾下划线的名称，用于特殊用途。 这样的例子有，__init__对象构造函数，或__call__ --- 它使得一个对象可以被调用。
    
    这些dunder方法通常被称为神奇方法 - 但Python社区中的许多人都不喜欢这种方法。
    
    最好避免在自己的程序中使用以双下划线（“dunders”）开头和结尾的名称，以避免与将来Python语言的变化产生冲突。

- #### 单下划线 _
    按照习惯，有时候单个独立下划线是用作一个名字，来表示某个变量是临时的或无关紧要的。例如，在下面的循环中，我们不需要访问正在运行的索引，我们可以使用“_”来表示它只是一个临时值：
    ```
    >>> for _ in range(32):
    ...    print('Hello, World.')
    ```
    你也可以在拆分(unpacking)表达式中将单个下划线用作“不关心的”变量，以忽略特定的值。 同样，这个含义只是“依照约定”，并不会在Python解释器中触发特殊的行为。 单个下划线仅仅是一个有效的变量名称，会有这个用途而已。在下面的代码示例中，我将汽车元组拆分为单独的变量，但我只对颜色和里程值感兴趣。 但是，为了使拆分表达式成功运行，我需要将包含在元组中的所有值分配给变量。 在这种情况下，“_”作为占位符变量可以派上用场：
    ```
    >>> car = ('red', 'auto', 12, 3812.4)
    >>> color, _, _, mileage = car

    >>> color
    'red'
    >>> mileage
    3812.4
    >>> _
    12
    ```
    除了用作临时变量之外，“_”是大多数Python REPL中的一个特殊变量，它表示由解释器评估的最近一个表达式的结果。这样就很方便了，比如你可以在一个解释器会话中访问先前计算的结果，或者，你是在动态构建多个对象并与它们交互，无需事先给这些对象分配名字：
    ```
    >>> 20 + 3
    23
    >>> _
    23
    >>> print(_)
    23

    >>> list()
    []
    >>> _.append(1)
    >>> _.append(2)
    >>> _.append(3)
    >>> _
    [1, 2, 3]
    ```
- #### 总结
    <img src="assets/20180927190455.png" width=550>
