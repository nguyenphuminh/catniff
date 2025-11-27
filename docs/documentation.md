# All APIs

Below is the specification for Catniff APIs. Note that undocumented APIs in the codebase are either unsafe or not ready for use other than internal.

## dtype

A value of type `dtype` is a string and can be `float64`, `float32`, `float16`, `int32`, `int16`, `int8`, `uint32`, `uint16`, or `uint8`.

## MemoryBuffer

A value of type `MemoryBuffer` can be `Float64Array`, `Float32Array`, `Float16Array`, `Int32Array`, `Int16Array`, `Int8Array`, `Uint32Array`, `Uint16Array`, or `Uint8Array`.

## TensorValue

Type `TensorValue` is either `number` or `TensorValue[]`, which means it represents either numbers or n-D number arrays.

## TensorOptions

`TensorOptions` is an interface that contains options/configurations of a tensor passed into the `Tensor` class constructor (more on that later). It includes:

* `shape?: number[]`
* `strides?: number[]`
* `offset?: number;`
* `numel?: number;`
* `grad?: Tensor`
* `requiresGrad?: boolean`
* `gradFn?: Function`
* `children?: Tensor[]`
* `device?: string;`
* `dtype?: dtype;`

## Tensor

### Constructor

```ts
constructor(value: TensorValue, options: TensorOptions = {})
```

### Properties

* `public value: MemoryBuffer`: Holds the tensor value/data as a `MemoryBuffer`, initialized by the `value` parameter but flattened and converted into a typed array with type specified in the `dtype` prop. If `value` is already of type `dtype`, it will use the value directly rather than copying.
* `public shape: number[]`: Holds the tensor shape, uses `options.shape` if provided, `Tensor.getShape(value)` otherwise.
* `public strides: number[]`: Holds the tensor strides, uses `options.strides` if provided, `Tensor.getStrides(this.shape)` otherwise.
* `public offset: number`: Holds the tensor storage offset, uses `options.offset` if provided, 0 otherwise.
* `public numel: number`: Holds the tensor tensor size (number of real elements, not this.value.length), uses `options.numel` if provided, `Tensor.shapeToSize(this.shape)` otherwise.
* `public grad?: Tensor`: Holds the tensor gradient, uses `options.grad` if provided, `undefined` otherwise to save memory.
* `public requiresGrad: boolean`: Choose whether to do gradient-related operations behind the scenes, uses `options.requiresGrad` if provided, `false` otherwise.
* `public gradFn: Function`: Called when computing gradient all over the DAG, used to feed gradient to its child tensors, uses `options.gradFn` if provided, `() => {}` otherwise.
* `public children: Tensor[]`: Holds its child tensors, will be used when computing gradient, uses `options.children` if provided, `[]` otherwise.
* `public device: string`: Holds the device the tensor is on, uses `options.device` if provided, `cpu` otherwise.
* `public dtype: dtype`: Holds the tensor's data type, uses `options.dtype` if provided, `float32` otherwise.
* `static training: boolean = false;`: Holds training flag, set to `true` while training to enable features like dropout, set to `false` while not to prevent unexpected behaviors.
* `static noGrad: boolean = false;`: Set to `true` to disable grad accumulation.
* `static createGraph: boolean = false;`: Preserves graph, set to `true` when computing nth-order derivative.
* `static backends: Map<string, Backend>`: Holds backends, scroll way down below to see what to do with this.

### Methods

All autograd-supported tensor arithmetic methods:

* `add(other: TensorValue | Tensor): Tensor`: Returns `this` added with `other` element-wise. If `other` is a `TensorValue`, it will be converted to a `Tensor` with `this.handleOther(other)`, and this rule will apply for other element-wise ops as well. Two tensors of different shapes and sizes will get broadcasted if they are compatible, or else the function will throw an error.
* `sub(other: TensorValue | Tensor): Tensor`: Returns `this` subtracted by `other` element-wise.
* `subtract(other: TensorValue | Tensor): Tensor`: Alias for `sub`.
* `mul(other: TensorValue | Tensor): Tensor`: Returns `this` multiplied with `other` element-wise.
* `multiply(other: TensorValue | Tensor): Tensor`: Alias for `mul`.
* `pow(other: TensorValue | Tensor): Tensor`: Returns `this` raised to the power of `other` element-wise.
* `div(other: TensorValue | Tensor): Tensor`: Returns `this` divided by `other` element-wise.
* `divide(other: TensorValue | Tensor): Tensor`: Alias for `div`.
* `remainder(other: TensorValue | Tensor): Tensor`: Returns remainder of `this` divided by `other` element-wise.
* `ge(other: TensorValue | Tensor): Tensor`: Returns `this` greater than or equal to `other` element-wise (1 if true, 0 if false).
* `greaterEqual(other: TensorValue | Tensor): Tensor`: Alias for `ge`.
* `le(other: TensorValue | Tensor): Tensor`: Returns `this` less than or equal to `other` element-wise (1 if true, 0 if false).
* `lessEqual(other: TensorValue | Tensor): Tensor`: Alias for `le`.
* `gt(other: TensorValue | Tensor): Tensor`: Returns `this` greater than `other` element-wise (1 if true, 0 if false).
* `greater(other: TensorValue | Tensor): Tensor`: Alias for `gt`.
* `lt(other: TensorValue | Tensor): Tensor`: Returns `this` less than `other` element-wise (1 if true, 0 if false).
* `less(other: TensorValue | Tensor): Tensor`: Alias for `lt`.
* `eq(other: TensorValue | Tensor): Tensor`: Returns `this` equal to `other` element-wise (1 if true, 0 if false).
* `equal(other: TensorValue | Tensor): Tensor`: Alias for `eq`.
* `ne(other: TensorValue | Tensor): Tensor`: Returns `this` not equal to `other` element-wise (1 if true, 0 if false).
* `notEqual(other: TensorValue | Tensor): Tensor`: Alias for `ne`.
* `logicalAnd(other: TensorValue | Tensor): Tensor`: Returns `this` logical and `other` element-wise (1 if both are 1, 0 otherwise).
* `logicalOr(other: TensorValue | Tensor): Tensor`: Returns `this` logical or `other` element-wise (1 if either are 1, 0 otherwise).
* `logicalXor(other: TensorValue | Tensor): Tensor`: Returns `this` logical xor `other` element-wise (1 if both are not the same bit, 0 otherwise).
* `logicalNot(): Tensor`: Returns logical not of `this` element-wise (1 if 0, 0 if 1).
* `bitwiseAnd(other: TensorValue | Tensor): Tensor`: Returns `this` bitwise and `other` element-wise.
* `bitwiseOr(other: TensorValue | Tensor): Tensor`: Returns `this` bitwise or `other` element-wise.
* `bitwiseXor(other: TensorValue | Tensor): Tensor`: Returns `this` bitwise xor `other` element-wise.
* `bitwiseNot(): Tensor`: Returns bitwise not of `this` element-wise.
* `bitwiseLeftShift(other: TensorValue | Tensor): Tensor`: Returns `this` bitwise left shift `other` element-wise.
* `bitwiseRightShift(other: TensorValue | Tensor): Tensor`: Returns `this` bitwise right shift `other` element-wise.
* `neg(): Tensor`: Returns negative of `this` element-wise.
* `negative(): Tensor`: Alias for `neg`.
* `reciprocal(): Tensor`: Returns reciprocal of `this` element-wise.
* `square(): Tensor`: Returns `this` squared element-wise.
* `abs(): Tensor`: Returns absolute of `this` element-wise.
* `absolute(): Tensor`: Alias for `abs`.
* `sign(): Tensor`: Returns sign of `this` element-wise.
* `sin(): Tensor`: Returns sin of `this` element-wise.
* `cos(): Tensor`: Returns cos of `this` element-wise.
* `tan(): Tensor`: Returns tan of `this` element-wise.
* `asin(): Tensor`: Returns asin of `this` element-wise.
* `arcsin(): Tensor`: Alias for `asin`.
* `acos(): Tensor`: Returns acos of `this` element-wise.
* `arccos(): Tensor`: Alias for `acos`.
* `atan(): Tensor`: Returns atan of `this` element-wise.
* `arctan(): Tensor`: Alias for `atan`.
* `atan2(other: TensorValue | Tensor): Tensor`: Returns arctan2 of `this` and `other` element-wise.
* `arctan2(other: TensorValue | Tensor): Tensor`: Alias for `atan2`.
* `sinh(): Tensor`: Returns sinh of `this` element-wise.
* `cosh(): Tensor`: Returns cosh of `this` element-wise.
* `asinh(): Tensor`: Returns asinh of `this` element-wise.
* `arcsinh(): Tensor`: Alias for `asinh`.
* `acosh(): Tensor`: Returns acosh of `this` element-wise.
* `arccosh(): Tensor`: Alias for `acosh`.
* `atanh(): Tensor`: Returns atanh of `this` element-wise.
* `arctanh(): Tensor`: Alias for `atanh`.
* `deg2rad(): Tensor`: Convert `this` degree to radian element-wise.
* `rad2deg(): Tensor`: Convert `this` radian to degree element-wise.
* `sqrt(): Tensor`: Returns square root of `this` element-wise.
* `rsqrt(): Tensor`: Returns reciprocal of square root of `this` element-wise.
* `exp(): Tensor`: Returns e raised to the power of `this` element-wise.
* `exp2(): Tensor`: Returns 2 raised to the power of `this` element-wise.
* `expm1(): Tensor`: Returns e raised to the power of `this` minus 1 element-wise.
* `log(): Tensor`: Returns natural log of `this` element-wise.
* `log2(): Tensor`: Returns log base 2 of `this` element-wise.
* `log10(): Tensor`: Returns log base 10 of `this` element-wise.
* `log1p(): Tensor`: Returns natural log of 1 plus `this` element-wise.
* `relu(): Tensor`: Returns relu of `this` element-wise.
* `leakyRelu(negativeSlope = 0.01): Tensor`: Returns leaky relu of `this` element-wise.
* `elu(alpha = 1): Tensor`: Returns elu of `this` element-wise.
* `selu(): Tensor`: Returns selu of `this` element-wise.
* `celu(alpha = 1): Tensor`: Returns celu of `this` element-wise.
* `sigmoid(): Tensor`: Returns sigmoid of `this` element-wise.
* `tanh(): Tensor`: Returns tanh of `this` element-wise.
* `softplus(): Tensor`: Returns softplus of `this` element-wise.
* `softsign(): Tensor`: Returns softsign of `this` element-wise.
* `silu(): Tensor`: Returns silu (swish) of `this` element-wise.
* `mish(): Tensor`: Returns mish of `this` element-wise.
* `gelu(approximate: string = "none"): Tensor`: Returns gelu of `this` element-wise. Use original gelu formula if `approximate` is `none`, use tanh approximation if set to `tanh`.
* `maximum(other: TensorValue | Tensor): Tensor`: Returns maximum between `this` and `other` element-wise.
* `minimum(other: TensorValue | Tensor): Tensor`: Returns minimum between `this` and `other` element-wise.
* `round(): Tensor`: Returns `this` rounded element-wise.
* `floor(): Tensor`: Returns `this` floored element-wise.
* `ceil(): Tensor`: Returns `this` ceiled element-wise.
* `trunc(): Tensor`: Returns `this` truncated element-wise.
* `fix(): Tensor`: Alias for `trunc`.
* `frac(): Tensor`: Returns fraction part of `this` element-wise.
* `clip(min: number, max: number): Tensor`: Returns value limited between `min` and `max`.
* `clamp(min: number, max: number): Tensor`: Alias for `clip`.
* `erf(): Tensor`: Returns error function with `this` as input element-wise.
* `erfc(): Tensor`: Returns complementary error function with `this` as input element-wise.
* `erfinv(): Tensor`: Returns inverse error function with `this` as input element-wise.
* `transpose(dim1: number, dim2: number): Tensor`: Returns transposition of a tensor from two provided dimensions.
* `t(): Tensor`: Returns transposition of a 2D tensor (matrix). If `this` is not 2D, it will throw an error.
* `permute(dims: number[]): Tensor`: Returns complete reposition of dims in a tensor.
* `isContiguous(): boolean`: Checks if tensor is contiguous.
* `contiguous(): Tensor`: Returns a new tensor, restructured from input to be contiguous.
* `reshape(newShape: number[]): Tensor`: Returns input, reshaped based on `newShape` provided.
* `view(newShape: number[]): Tensor`: Returns input, reshaped based on `newShape` provided. This is different from reshape in that it will only return a view (does not allocate new mem) of the original tensor and throws an error if the tensor can not be reshaped by just modifying the metadata, while reshape will force it to be contiguous if it is not compatible, thus using more mem without an error.
* `flatten(startDim = 0, endDim = -1): Tensor`: Returns input flattened from `startDim` to `endDim`.
* `index(indices: Tensor | TensorValue): Tensor`: Returns a new tensor with items indexed from `this` tensor. For example, if `this` has shape `[3,4,5]`, and `indices` is a scalar, the result will have shape `[4,5]`, and if `indices` has shape `[2,3]`, the result will have shape `[2,3,4,5]`. There is also `indexWithArray` but `indices` are only of type `number[]`.
* `slice(ranges: number[][]): Tensor`: Slice a child tensor. Each range applies to each dimension and has a form of `[start, end, step]` where `start` is `0` by default; `end` is max dim size; and `step` is 1 by default.
* `chunk(chunks: number, dim = 0): Tensor[]`: Returns a `chunks` number of chunks split from `this`, at dimension `dim`.
* `expand(newShape: number[]): Tensor`: Returns a new tensor expanded to `newShape`.
* `cat(other: Tensor | TensorValue, dim = 0): Tensor`: Concatenate `this` tensor with `other` tensor along the specified dimension `dim`. 
* `dot(other: TensorValue | Tensor): Tensor`: Returns the vector dot product of `this` and `other` 1D tensors (vectors). If the two are not 1D, it will throw an error.
* `mm(other: TensorValue | Tensor): Tensor`: Returns the matrix multiplication of `this` and `other` 2D tensors (matrices). If the two are not 2D, it will throw an error.
* `mv(other: TensorValue | Tensor): Tensor`: Returns the matrix multiplication of `this` 2D tensor (matrix) and `other` 1D tensor (vector). Basically if `other` is of size n, it will be reshaped into an nx1 matrix. If `this` is not 2D and `other` is not 1D, it will throw an error.
* `bmm(other: TensorValue | Tensor): Tensor`: Returns the batched matrix multiplication of `this` and `other` 3D tensors (batches of matrices). If the two are not 3D, it will throw an error.
* `matmul(other: TensorValue | Tensor): Tensor`: Returns the matrix multiplication of `this` and `other`. If both are 1D then `dot` is used; if both are 2D then `mm` is used; if `this` is 2D and `other` is 1D then `mv` is used; if `this` is 1D and `other` is 2D then a size-1 dimension will be padded into `this` to do `mm`, then the padded dimension will be removed; if at least one is nD, then output is broadcasted and then a batched matmul is done on two last axes.
* `squeeze(dims?: number[] | number): Tensor`: Returns a new tensor with size-1 dims squeezed out. If `dims` is `undefined`, all size-1 dims are squeezed out. If `dims` is a number/number array, it will squeeze out dimensions at that/those positions. If a specified dimension is not size-1, it will throw an error.
* `unsqueeze(dims?: number[] | number): Tensor`: Returns a new tensor with size-1 dims pushed into specified positions. If `dims` is `undefined`, a tensor with same value, shape, and strides will be returned. If `dims` is a number/number array, it will push size-1 dimensions into that/those positions.
* `sum(dims?: number[] | number, keepDims: boolean = false): Tensor`: Returns a new tensor with axes summed. If `dims` is `undefined`, all axes will be summed into a scalar. If `dims` is a number/number array, it will sum dimensions at that/those positions. If `keepDims` is `true`, then the size-1 dimensions after summation will be kept, discarded otherwise.
* `prod(dims?: number[] | number, keepDims: boolean = false): Tensor`: Returns a new tensor with axes reduced to their products.
* `mean(dims?: number[] | number, keepDims: boolean = false): Tensor`: Returns a new tensor with axes reduced to their means.
* `max(dims?: number[] | number, keepDims: boolean = false): Tensor`: Returns a new tensor with axes reduced to their maximums.
* `min(dims?: number[] | number, keepDims: boolean = false): Tensor`: Returns a new tensor with axes reduced to their minimums.
* `any(dims?: number[] | number, keepDims: boolean = false): Tensor`: Returns a new tensor with axes reduced to 1 if a value in a dim is 1, 0 otherwise.
* `all(dims?: number[] | number, keepDims: boolean = false): Tensor`: Returns a new tensor with axes reduced to 1 if all values in a dim are 1, 0 otherwise.
* `var(dims?: number[] | number, keepDims: boolean = false): Tensor`: Returns a new tensor with axes reduced to their variances.
* `std(dims?: number[] | number, keepDims: boolean = false): Tensor`: Returns a new tensor with axes reduced to their standard deviations.
* `softmax(dim: number = -1): Tensor`: Apply numerically stable softmax on the specified dimension.
* `softmin(dim: number = -1): Tensor`: Apply numerically stable softmin on the specified dimension.
* `dropout(rate: number): Tensor`: Apply dropout with `rate`, only works when `Tensor.training` is `true`.
* `triu(diagonal=0): Tensor`: Get the upper triangular part with respect to main diagonal (the lower part is set to 0).
* `tril(diagonal=0): Tensor`: Get the lower triangular part with respect to main diagonal (the upper part is set to 0).
* `maskedFill(mask: Tensor | TensorValue, value: number): Tensor`: Fill specific positions of this tensor with a `value` through a `mask` (1 for fill, 0 for unchanged).

Here are commonly used utilities:

* `backward(options: { zeroGrad?: boolean } = {})`: Calling this will recursively accumulate gradients of nodes in the DAG you have built, with the tensor you call backward() on as the root node for gradient computation. Note that this will assume the gradient of the top node to be a tensor of same shape, filled with 1, and it will zero out the gradients of child nodes before calculation if not explicitly specified in `options.zeroGrad`.
* `cast(dtype: dtype): Tensor`: Return a new tensor casted to `dtype`.
* `val(): TensorValue`: Returns the raw nD array/number form of the tensor.
* `detach(): Tensor`: Returns a view of the tensor with requiresGrad changed to `false` and detaches from DAG.
* `clone(): Tensor`: Returns a copy of the tensor (with new data allocation) and keeps grad connection.
* `replace(other: Tensor | TensorValue): Tensor`: Returns this tensor with value replaced with the value of another tensor.
* `to(device: string): Tensor`: Returns a new tensor with the same value as this tensor, but on a different device.
* `static full(shape: number[], num: number, options: TensorOptions = {}): Tensor`: Returns a new tensor with provided `shape`, filled with `num`, configured with `options`.
* `static fullLike(tensor: Tensor, num: number, options: TensorOptions = {}): Tensor`: Returns a new tensor of same shape and strides as `tensor`, filled with `num`, configured with `options`.
* `static ones(shape?: number[], options: TensorOptions = {}): Tensor`: Returns a new tensor with provided `shape`, filled with 1, configured with `options`.
* `static onesLike(tensor: Tensor, options: TensorOptions = {}): Tensor`: Returns a new tensor of same device, shape, and strides as `tensor`, filled with 1, configured with `options`.
* `static zeros(shape?: number[], options: TensorOptions = {}): Tensor`: Returns a new tensor with provided `shape`, filled with 0, configured with `options`.
* `static zerosLike(tensor: Tensor, options: TensorOptions = {}): Tensor`: Returns a new tensor of same device, shape, and strides as `tensor`, filled with 0, configured with `options`.
* `static rand(shape?: number[], options: TensorOptions = {}): Tensor`: Returns a new tensor with provided `shape`, filled with a random number with uniform distribution from 0 to 1, configured with `options`.
* `static randLike(tensor: Tensor, options: TensorOptions = {}): Tensor`: Returns a new tensor of same device, shape, and strides as `tensor`, filled with a random number with uniform distribution from 0 to 1, configured with `options`.
* `static randn(shape?: number[], options: TensorOptions = {}): Tensor`: Returns a new tensor with provided `shape`, filled with a random number with normal distribution of mean=0 and stddev=1, configured with `options`.
* `static randnLike(tensor: Tensor, options: TensorOptions = {}): Tensor`: Returns a new tensor of same device, shape, and strides as `tensor`, filled with a random number with normal distribution of mean=0 and stddev=1, configured with `options`.
* `static randint(shape: number[], low: number, high: number, options: TensorOptions = {}): Tensor`: Returns a new tensor with provided `shape`, filled with a random integer between low and high, configured with `options`.
* `static randintLike(tensor: Tensor, low: number, high: number, options: TensorOptions = {}): Tensor`: Returns a new tensor of same device, shape, and strides as `tensor`, filled with a random integer between low and high, configured with `options`.
* `static randperm(n: number, options: TensorOptions = {}): Tensor`: a new tensor filled with a random permutation of integers from 0 to `n-1`, configured with `options`.
* `static normal(shape: number[], mean: number, stdDev: number, options: TensorOptions = {}): Tensor`: Returns a new tensor with provided `shape`, filled with a random number with normal distribution of custom `mean` and `stdDev`, configured with `options`.
* `static uniform(shape: number[], low: number, high: number, options: TensorOptions = {}): Tensor`: Returns a new tensor with provided `shape`,  filled with a random number with uniform distribution from `low` to `high`, configured with `options`.
* `static eye(n: number, m: number = n, options: TensorOptions = {}): Tensor`: Returns a 2D tensor (matrix of size `nxm`) with its main diagonal filled with 1s and others with 0s, configured with `options`.
* `static linspace(start: number, stop: number, steps: number, options: TensorOptions = {}): Tensor`: Returns a new 1D tensor from a range evenly spaced out with a given amount of steps, configured with `options`.
* `static arange(start: number, stop?: number, step = 1, options: TensorOptions = {}): Tensor`: Returns a new 1D tensor from a range incrementing with `step`, configured with `options`. If `stop` is not provided, `start` will be `0` and `stop` will be the original `start`.


Here are utilities (that might be deleted in the future) that you probably won't have to use but they might come in handy:

* `static flattenValue(tensorValue: TensorValue): ArrayLike<number>`: Used to flatten an n-D array to 1D, numbers will be converted into size-1 arrays.
* `static getShape(tensor: TensorValue): number[]`: Used to get shape (size of each dimension) of an n-D array as a number array.
* `static getStrides(shape: number[]): number[]`: Used to get strides of tensor from its shape. Strides are needed internally because they are steps taken to get a value at each dimension now that the tensor has been flatten to 1D.
* `static padShape`: Used to pad shape and strides of two tensors to be of same number of dimensions.
    * args:
        * `stridesA: number[]`: Strides of the first tensor.
        * `stridesB: number[]`: Strides of the second tensor.
        * `shapeA: number[]`: Shape of the first tensor.
        * `shapeB: number[]`: Shape of the second tensor.
    * returns: A tuple of `(newStridesA, newStridesB, newShapeA, newShapeB)` with type `[number[], number[], number[], number[]]`.
* `static broadcastShapes(shapeA: number[], shapeB: number[]): number[]`: Returns the new shape broadcasted from `shapeA` and `shapeB`. Basically if one shape's dimension is of size n, and other shape's corresponding dimension if of size n or 1, then the new shape's corresponding dimension is n, otherwise throw an error. For example `[1,2,3]` and `[4,1,3]` would be `[4,2,3]` after broadcasting.
* `static indexToCoords(index: number, strides: number[]): number[]`: Convert an index of an 1D array to coordinates (indices) of an nD array, based on the nD array's `strides`.
* `static coordsToIndex(coords: number[], strides: number[]): number`: Convert coordinates (indices) of an nD array to an index of an 1D array, based on the nD array's `strides`.
* `static coordsToUnbroadcastedIndex(coords: number[], shape: number[], strides: number[]): number`: Convert coordinates (indices) of an unbroadcasted nD array to an index of an 1D array, based on the nD array's `shape` and `strides`. Basically the same as above but coordinates of dimensions with size 1 are forced to be 0.
* `static shapeToSize(shape: number[]): number`: Convert shape into 1D array size.
* `static normalizeDims(dims: number[], numDims: number): number[]`: Convert negative dims to normal and check if out of bound.
* `static elementWiseAB(tA: Tensor, tB: Tensor, op: (tA: number, tB: number) => number): Tensor`: Perform a custom element-wise `op` between two tensors, returns a new tensor that holds the result.
* `static elementWiseSelf(tA: Tensor, op: (tA: number) => number): Tensor`: Perform a custom element-wise `op` on a tensor, returns a new tensor that holds the result.
* `elementWiseABDAG`: Perform a custom element-wise op between this tensor with another tensor. If `this` or `other` have `requiresGrad` as `true`, it will build a DAG node for future gradient computation.
    * args:
        * `other: TensorValue | Tensor`: The other tensor.
        * `op: (a: number, b: number) => number`: The custom op to do element-wise.
        * `thisGrad: (self: Tensor, other: Tensor, outGrad: Tensor) => Tensor = () => new Tensor(0)`: Custom gradient for `this` tensor if `this.requiresGrad` is `true`, returns a tensor that will be assigned to `this.grad`. Note that `self` represents `this` tensor, `other` represents the `other` tensor above, and `outGrad` represents the upstream gradient, but all of these tensors have all gradient-related operations disabled and are not the original tensors.
        * `otherGrad: (self: Tensor, other: Tensor, outGrad: Tensor) => Tensor = () => new Tensor(0)`: Same as above but assigned to `other.grad`.
    * returns: A new `Tensor`. If `this.requiresGrad` is `true`, then `this` will be a child of the new tensor, same with `other`.
* `elementWiseSelfDAG`: Perform a custom element-wise op on a tensor. If `this.requiresGrad` is `true`, it will build a DAG node for future gradient computation.
    * args:
        * `op: (a: number) => number`: The custom op to do element-wise.
        * `thisGrad: (self: Tensor, outGrad: Tensor) => Tensor = () => new Tensor(0)`: Custom gradient for `this` tensor if `this.requiresGrad` is `true`, returns a tensor that will be assigned to `this.grad`. Note that `self` represents `this` tensor, and `outGrad` represents the upstream gradient, but all of these tensors have all gradient-related operations disabled and are not the original tensors.
    * returns: A new `Tensor`. If `this.requiresGrad` is `true`, then `this` will be a child of the new tensor`.
* `handleOther(value: TensorValue | Tensor): Tensor`: Returns the argument if it already is a `Tensor` instance, otherwise create a new `Tensor` instance with `value` as input that is on the same device as `this`. It will throw an error if the param is a tensor that is not on the same device as `this`.
* `static addGrad(tensor: Tensor, accumGrad: Tensor)`: Add to the `grad` prop of a tensor. It can handle broadcasted shapes and make `accumGrad` fit `tensor`'s shape.

## SGDOptions

`SGDOptions` is an interface that contains options/configurations of an SGD optimizer passed into the `Optim.SGD` class constructor (more on that later). It includes:

* `lr?: number`
* `momentum?: number`
* `dampening?: number`
* `weightDecay?: number`
* `nesterov?: boolean`

## Optim.BaseOptimizer / BaseOptimizer (abtract class)

### Constructor

```ts
constructor(params: Tensor[], options?: SGDOptions)
```

### Properties

* `public params: Tensor[]`: Holds the params to be optimized, initialized with the `params` argument mentioned above.

### Methods

* `zeroGrad()`: Set the `grad` property of each param in `this.params` to `Tensor.zerosLike(param)`.

## Optim.SGD extends Optim.BaseOptimizer / SGD

### Constructor

```ts
constructor(params: Tensor[], options?: SGDOptions)
```

### Properties

* `public momentumBuffers: Map<Tensor, Tensor> = new Map()`: Holds the current momentum buffer of each param, updated per optimization iteration if `this.momentum` is not `0`.
* `public lr: number`: Holds the learning rate, uses `options.lr` if available, `0.001` otherwise.
* `public momentum: number`: Holds the momentum, uses `options.momentum` if available, `0` otherwise.
* `public dampening: number`: Holds the dampening, uses `options.dampening` if available, `0` otherwise.
* `public weightDecay: number`: Holds the weight decay rate, uses `options.weightDecay` if available, `0` otherwise.
* `public nesterov: boolean`: Chooses whether to use nesterov (NAG) optimization or not, uses `options.nesterov` if available, `false` otherwise.

### Methods

* `step()`: Perform one SGD iteration and update values of parameters in-place.

## AdamOptions

`AdamOptions` is an interface that contains options/configurations of an Adam optimizer passed into the `Optim.Adam` class constructor (more on that later). It includes:

* `lr?: number`
* `betas?: [number, number]`
* `eps?: number`
* `weightDecay?: number`

## Optim.Adam extends Optim.BaseOptimizer / Adam

### Constructor

```ts
constructor(params: Tensor[], options?: AdamOptions)
```

### Properties

* `public momentumBuffers: Map<Tensor, Tensor> = new Map()`: Holds the current momentum (first moment) buffer of each param.
* `public velocityBuffers: Map<Tensor, Tensor> = new Map()`: Holds the current velocity (second moment) buffer of each param.
* `public lr: number`: Holds the learning rate, uses `options.lr` if available, `0.001` otherwise.
* `public betas: [number, number]`: Holds the momentum, uses `options.betas` if available, `[0.9, 0.999]` otherwise.
* `public eps: number`: Holds the dampening, uses `options.eps` if available, `1e-8` otherwise.
* `public weightDecay: number`: Holds the weight decay rate, uses `options.weightDecay` if available, `0` otherwise.

### Methods

* `step()`: Perform one Adam iteration and update values of parameters in-place.

## AdamWOptions

`AdamWOptions` is an interface that contains options/configurations of an AdamW optimizer passed into the `Optim.AdamW` class constructor (more on that later). It includes:

* `lr?: number`
* `betas?: [number, number]`
* `eps?: number`
* `weightDecay?: number`

## Optim.AdamW extends Optim.BaseOptimizer / AdamW

### Constructor

```ts
constructor(params: Tensor[], options?: AdamWOptions)
```

### Properties

* `public momentumBuffers: Map<Tensor, Tensor> = new Map()`: Holds the current momentum (first moment) buffer of each param.
* `public velocityBuffers: Map<Tensor, Tensor> = new Map()`: Holds the current velocity (second moment) buffer of each param.
* `public lr: number`: Holds the learning rate, uses `options.lr` if available, `0.001` otherwise.
* `public betas: [number, number]`: Holds the momentum, uses `options.betas` if available, `[0.9, 0.999]` otherwise.
* `public eps: number`: Holds the dampening, uses `options.eps` if available, `1e-8` otherwise.
* `public weightDecay: number`: Holds the weight decay rate, uses `options.weightDecay` if available, `0.01` otherwise.

### Methods

* `step()`: Perform one AdamW iteration and update values of parameters in-place.

## nn.Linear / Linear

### Constructor

```ts
constructor(
    inFeatures: number,
    outFeatures: number,
    bias: boolean = true,
    device?: string,
    dtype?: dtype
)
```

### Properties

* `public weight: Tensor`: Weight of linear layer.
* `public bias?: Tensor`: Bias of linear layer.

### Methods

* `forward(input: Tensor | TensorValue): Tensor`: Forward-pass `input` through the linear layer.

## nn.RNNCell / RNNCell

### Constructor

```ts
constructor(
    inputSize: number,
    hiddenSize: number,
    bias: boolean = true,
    device?: string,
    dtype?: dtype
)
```

### Properties

* `public weightIH: Tensor`: Input weight.
* `public weightHH: Tensor`: Hidden weight.
* `public biasIH?: Tensor`: Input bias.
* `public biasHH?: Tensor`: Hidden bias.

### Methods

* `forward(input: Tensor | TensorValue, hidden: Tensor | TensorValue): Tensor`: Forward-pass `input` through the recurrent cell, returning the new hidden state.

## nn.GRUCell / GRUCell

### Constructor

```ts
constructor(
    inputSize: number,
    hiddenSize: number,
    bias: boolean = true,
    device?: string,
    dtype?: dtype
)
```

### Properties

* `public weightIR: Tensor`: Weight of input in reset gate.
* `public weightIZ: Tensor`: Weight of input in update gate.
* `public weightIN: Tensor`: Weight of input in candidate gate.
* `public weightHR: Tensor`: Weight of hidden state in reset gate.
* `public weightHZ: Tensor`: Weight of hidden state in update gate.
* `public weightHN: Tensor`: Weight of hidden state in candidate gate.
* `public biasIR?: Tensor`: Bias of input in reset gate.
* `public biasIZ?: Tensor`: Bias of input in update gate.
* `public biasIN?: Tensor`: Bias of input in candidate gate.
* `public biasHR?: Tensor`: Bias of hidden state in reset gate.
* `public biasHZ?: Tensor`: Bias of hidden state in update gate.
* `public biasHN?: Tensor`: Bias of hidden state in candidate gate.

### Methods

* `forward(input: Tensor | TensorValue, hidden: Tensor | TensorValue): Tensor`: Forward-pass `input` through the GRU cell, returning the new hidden state.

## nn.LSTMCell / LSTMCell

### Constructor

```ts
constructor(
    inputSize: number,
    hiddenSize: number,
    bias: boolean = true,
    device?: string,
    dtype?: dtype
)
```

### Properties

* `public weightII: Tensor`: Weight of input in input gate.
* `public weightIF: Tensor`: Weight of input in forget gate.
* `public weightIG: Tensor`: Weight of input in candidate cell gate.
* `public weightIO: Tensor`: Weight of input in output gate.
* `public weightHI: Tensor`: Weight of hidden state in input gate.
* `public weightHF: Tensor`: Weight of hidden state in forget gate.
* `public weightHG: Tensor`: Weight of hidden state in candidate cell gate.
* `public weightHO: Tensor`: Weight of hidden state in output gate.
* `public biasII?: Tensor`: Bias of input in input gate.
* `public biasIF?: Tensor`: Bias of input in forget gate.
* `public biasIG?: Tensor`: Bias of input in candidate cell gate.
* `public biasIO?: Tensor`: Bias of input in output gate.
* `public biasHI?: Tensor`: Bias of hidden state in input gate.
* `public biasHF?: Tensor`: Bias of hidden state in forget gate.
* `public biasHG?: Tensor`: Bias of hidden state in candidate cell gate.
* `public biasHO?: Tensor`: Bias of hidden state in output gate.

### Methods

* `forward(input: Tensor | TensorValue, hidden: Tensor | TensorValue, cell: Tensor | TensorValue): [Tensor, Tensor]`: Forward-pass `input` through the LSTM cell, returning the new hidden state and cell state.

## nn.LayerNorm / LayerNorm

### Constructor

```ts
constructor(
    normalizedShape: number | number[],
    eps: number = 1e-5,
    elementwiseAffine: boolean = true,
    bias: boolean = true,
    device?: string,
    dtype?: dtype
)
```

### Properties

* `public weight?: Tensor`: Weight to scale, available if `elementwiseAffine` is `true`.
* `public bias?: Tensor`: Bias to scale, available if `elementwiseAffine` and `bias` are `true`.
* `public eps: number`: Basically just `eps` from the constructor.
* `public normalizedShape: number[]`: Basically just `normalizedShape` from the constructor, padded into an array if needed.

### Methods

* `forward(input: Tensor): Tensor`: Apply layer norm on input tensor.

## nn.RMSNorm / RMSNorm

### Constructor

```ts
constructor(
    normalizedShape: number | number[],
    eps: number = 1e-5,
    elementwiseAffine: boolean = true,
    device?: string,
    dtype?: dtype
)
```

### Properties

* `public weight?: Tensor`: Weight to scale, available if `elementwiseAffine` is `true`.
* `public eps: number`: Basically just `eps` from the constructor.
* `public normalizedShape: number[]`: Basically just `normalizedShape` from the constructor, padded into an array if needed.

### Methods

* `forward(input: Tensor): Tensor`: Apply RMS norm on input tensor.

## nn.Embedding / Embedding

### Constructor

```ts
constructor(
    numEmbeddings: number,
    embeddingDim: number,
    device?: string,
    dtype?: dtype
)
```

### Properties

* `public weight: Tensor`: Weight to look up from, initialized with `Tensor.randn` with shape `[numEmbeddings, embeddingDim]`, on the specified `device`.

### Methods

* `forward(input: Tensor | TensorValue): Tensor`: Perform a lookup from the weight.

## nn.MultiheadAttention / MultiHeadAttention

### Constructor

```ts
constructor(
    embedDim: number,
    numHeads: number,
    dropout = 0,
    bias = true,
    device?: string,
    dtype?: dtype
)
```

### Properties

* `public qProjection: Linear`: A linear projection layer for queries, initialized with `new nn.Linear(embedDim, embedDim, bias, device)`.
* `public kProjection: Linear`: A linear projection layer for keys, initialized with `new nn.Linear(embedDim, embedDim, bias, device)`.
* `public vProjection: Linear`: A linear projection layer for values, initialized with `new nn.Linear(embedDim, embedDim, bias, device)`.
* `public oProjection: Linear`: A linear projection layer for outputs, initialized with `new nn.Linear(embedDim, embedDim, bias, device)`.
* `public embedDim: number`: Embedding dimension, from the `embedDim` param.
* `public numHeads: number`: Number of attention heads, from the `numHeads` param.
* `public headDim: number`: Dimension of a head, which is just `Math.floor(embedDim / numHeads)`.
* `public dropout: number`: Dropout rate, from the `dropout` param.

### Methods

* Forward pass:
```js
forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    needWeights = true,
    attnMask?: Tensor,
    averageAttnWeights = true
): [Tensor, Tensor | undefined]
```

## nn.state

### Methods

* `getParamemters(model: any, visited: WeakSet<object> = new WeakSet()): Tensor`: Collect all parameters (tensors) used in a model.
* `moveParameters(model: any, device: string): void`: Collect all parameters (tensors) used in a model and move it to another device.
* `getStateDict(model: any, prefix: string = "", visited: WeakSet<object> = new WeakSet()): StateDict`: Get Torch-style dictionary (object) of model's state (StateDict is just a flat object).
* `loadStateDict(model: any, stateDict: StateDict, prefix: string = "", visited: WeakSet<object> = new WeakSet()): void`: Load a model's params into another model through a given `stateDict`.

## StateDict

`StateDict` is just an object with string keys and values of type `any`.

## LRScheduler.StepLR

### Constructor

```ts
constructor(
    optimizer: BaseOptimizer,
    stepSize: number,
    gamma = 0.1,
    lastEpoch = -1
)
```

### Properties

* `public optimizer: BaseOptimizer;`: Holds the optimizer to get LR from, initialized with the `optimizer` param.
* `public stepSize: number;`: Holds the number of steps before an LR update, initialized with the `stepSize` param.
* `public gamma: number;`: Holds a number to multiply into LR , initialized with the `gamma` param.
* `public lastEpoch: number;`: Holds the last epoch, initialized with the `lastEpoch` param.
* `public baseLR: number;`: Holds the original LR of optimizer.

### Methods

* `step(epoch?: number)`: Apply a scheduler run one time. If an `epoch` is passed, it will replace the current `lastEpoch` prop.

# Custom backend

## Loading a custom backend

You can load a custom backend using:

```js
Tensor.backends.set("device_name", backend);
```

Then Catniff will use the backend's ops on tensors that have moved to `device_name`.

## Building a custom backend

There are two things to keep in mind when building your own custom backend - tensors and tensor ops.

For tensor values (`someTensor.value` for example), you should create a custom `Proxy` of a normal array, with its getter and setter targeting where the data was originally stored (using N-API to wrap C++ APIs for example) for compatibility with real JS number arrays. But of course this is only for compatibility, your tensor ops should use the original data for computation, so you should probably store a memory address/pointer of the original data in this proxy for your ops to know which tensor data to work with.

For tensor ops, you can reimplement whatever ops you want, but you should implement all methods that directly transform the data, `add` or `matmul` for example, not ops that just create new shapes and strides like `squeeze` or `transpose`. To be more specific, you can have a look at the ops in `./src/core.ts`, and whatever ops that return a new tensor with the same device as input do not need to be reimplemented, because those ops only modify metadata and does not read from or write to memory. Others if not implemented can either break or be very slow.

Other than that, you must create two methods for your backend, `transfer` for tensor transfer from another device to this device, `create` for doing that in-place. Here is an example (pay attention to the comments):
```js
const backend = {
    transfer(tensor) {
        // Create a new tensor object, reassign the ops, and do something to move to device here
        // ...

        // Reassign "to" to move from this device to another device
        tensor.to = function(device) {
            // Do something here

            // A backend does not exist for cpu, so you have to reimplement a way to move back to cpu
            if (device === "cpu") {
                // Create a new on-cpu array
                tensor.value = something;
                // Change device to cpu
                tensor.device = "cpu";
                // Reassign original ops:
                tensor.add = Tensor.prototype.add;
                // The rest of the code goes here
            }

            // Call the transfer method of another backend
            const backend = Tensor.backends.get(device);

            if (backend && backend.transfer) {
                return backend.transfer(this);
            }

            throw new Error(`No device found to transfer tensor to or a handler is not implemented for device.`);
        }

        // Reassign "to_" to move from this device to another device in-place
        tensor.to_ = function(device) {
            // ...

            // Same as to, but now use create:
            const backend = Tensor.backends.get(this.device);

            if (backend && backend.create) {
                backend.create(this);
                return this;
            }

            throw new Error(`No device found to transfer tensor to or a handler is not implemented for device.`);
        }

        return tensor;
    }

    create(tensor) {
        // Do the same as above, but modify into the tensor object directly
    }
}
```

