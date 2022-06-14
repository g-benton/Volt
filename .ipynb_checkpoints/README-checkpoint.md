# Volt
Public Implementation of 
*Volatility Based Kernels and Moving Average Means for Accurate Forecasting with Gaussian Processes* [link] 

by [Gregory Benton](https://g-benton.github.io/), [Wesley Maddox](https://wjmaddox.github.io), and [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).

Please cite our work if you find it useful:

```
@inproceedings{benton2022volatility,
  title={olatility Based Kernels and Moving Average Means for Accurate Forecasting with Gaussian Processes},
  author={Benton, Gregory and Maddox, Wesley and Wilson, Andrew Gordon Gordon},
  booktitle={International Conference on Machine Learning},
  year={2022},
  organization={PMLR}
}
```

![Overview of the Volt modeling pipeline](./figs/ret-vol-px.jpg)

## Explanatory Notebook

To see an overview of how to use Volt with synthetically generated code, see the `Example` notebook which walks through how the code is organized step by step.

## Experiments

The two core experimental settings from the paper involve modeling historical wind speeds and stock prices. The code to run these experiments with example commands is in the `experiments` folder.