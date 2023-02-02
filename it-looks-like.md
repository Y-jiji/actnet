```rust
// FnHandle
let a = act! ( a : b : [i, k] : sum { a[i, j] * b[j, k] : j } );

struct FnHandle<A, B> {
}

let device = Device::new();

// compile 是什么类型? 只能是 dyn FnDeviceHandle 类型, 因为函数的行为各种各样, 也有不同的存储数据的方式, 只能通过一个界面操作
let handle = device.compile(a);

// 一个handle只能用提供ownership的方式被调用, 最终它勾连了device当中的一个函数, 也可以理解为handle是结果的builder
let result = handle
    .call(input_a)
    .call(input_b)
    .compute()
;

// 导数/梯度的糟糕之处在于它们都是对表达式来说的, 而一个表达式可以有多个自由变元, 从而未必是纯函数. 
// 只能对值是标量的一个表达式求导, 用一个Tree类型的binder把自由变元全部绑定. 
// 用这个办法就能算出 gradient 了, 虽然有点拙劣, 但它的泛类型是确定的. 
let handle = device.to_grad(a);

// 把parameter全部列出来也不是一个理想的方案, 它会使得代码堆积得很冗长. 
// 而且寄希望于模型开发者搞清楚自己使用了多少参数也不现实. 
// 因此在这套系统上, 还要构建一个特别强调易用性的方案来给专注于模型本身的开发者使用. 
struct AddModule {
    param: Tensor<f32, Shape![1,2,3]>
}

// sum { (slv a b)[i] * a[i, j] : i } = b[j]

impl nn::Module for AddModule {
    // they are just place holders, real computation doesn't happen here
    type Input = Tensor<f32, Shape![1,2,3]>;
    // virtual tensor
    type Output = Tensor<f32, Shape![1,2,3]>;
    fn new() -> Self {
        AddModule { param: Tensor::zeros() }
    }
    fn forward(&self, input: Self::Input) -> Self::Output {
        let a = input.f32().dbg();
        let b = self.param.dbg();
        act!( [i, j, k] : a[i, j, k] + b[i, j, k] ).into()
    }
    fn parameter(&self) -> ParameterTree {
        ParameterTree::leaf(self.param)
    }
}

// 模型计算出的结果是带标记的表达式
struct ConvModule {
    layer_1: AddModule,
    layer_2: AddModule,
    layer_3: OutModule,
}

let device = DeviceCUDA::new(0);
let model = ConvModule::new().compile_to(device);
let loss = model.forward(x);
println!("{}", loss.item());
let grad = loss.backward();
model.update(grad);
println!("{}", model.parameter());

// 使用这个库的人不止是模型的开发者, 同时还有设备代码的开发者. 
// 为了让计算设备的开发者写得开心, 把ir做得漂亮是非常明智的. 
```