# Dataset Access

## Dataset Facts

| Property | Value |
|---|---|
| Dataset | Fashion-MNIST, a fashion product image dataset from Zalando Research |
| Task | Class-conditional generative modeling. The model is a conditional DCGAN that uses one of the `10` class labels. |
| Images | `70,000` grayscale images. Each image has a resolution of `28x28` pixels. The images come from Zalando retail product photographs. |
| Classes | `10` classes, with `7,000` images in each class |
| Training split | `60,000` images, with `6,000` images per class. The files use the IDX format. |
| Test split | `10,000` images, with `1,000` images per class. The files use the IDX format. |
| License | MIT (the upstream Fashion-MNIST repository by Zalando Research). This license permits academic use, redistribution, commercial use, and the creation of derivative generated samples under its terms. |

## Source

The official Fashion-MNIST dataset is available at:

`https://github.com/zalandoresearch/fashion-mnist`

The repository provides four `.gz` download archives in the IDX format.

The `torchvision.datasets.FashionMNIST` class provides access to these files.

## Access in the Notebook

The notebook loads the dataset from the local directory `data/FashionMNIST/raw/` at the repository root.

The load command is:

`datasets.FashionMNIST(root="data", download=False)`

This configuration uses local files first. The directory contains four `.gz` download archives and four extracted IDX files. There are `8` files in total.

If the local files are not available, the loader uses `download=True`. This option downloads the same public archives and extracts them to the same directory.

You do not need to download the files manually.

After the data loading cells, the notebook runs a hash-check cell. This cell checks the file checksums before training starts.

## Integrity

The hash values below come from the files in `data/FashionMNIST/raw/`.

The first download took place on `2026-10-10`. The notebook hash-check cell and a command-line check verified the files again on `2026-11-06`.

A new download with the `download=True` option on `2026-11-07` produced the same hash values.

Before training starts, the notebook checks the SHA256 checksums of all four download archives.

These checksums confirm the integrity and origin of the compressed archives as distributed.

The torchvision loader reads the extracted IDX files. The notebook does not calculate separate hashes for these extracted files.

The package does not include dataset files. Therefore, these hash values document the downloaded files. They do not represent the hash values of a new download.

| Archive | SHA256 | Size (bytes) |
|---|---|---|
| `train-images-idx3-ubyte.gz` | `3AEDE38D61863908AD78613F6A32ED271626DD12800BA2636569512369268A84` | `26,421,880` |
| `train-labels-idx1-ubyte.gz` | `A04F17134AC03560A47E3764E11B92FC97DE4D1BFAF8BA1A3AA29AF54CC90845` | `29,515` |
| `t10k-images-idx3-ubyte.gz` | `346E55B948D973A97E58D2351DDE16A484BD415D4595297633BB08F03DB6A073` | `4,422,102` |
| `t10k-labels-idx1-ubyte.gz` | `67DA17C76EAFFCA5446C3361AAAB5C3CD6D1C2608764D35DFB1850B086BF8DD5` | `5,148` |

The Fashion-MNIST README provides MD5 checksums for the same four archives.

The local files match these MD5 values. The SHA256 values above were calculated from the same files.

| Archive | MD5 (Fashion-MNIST README) |
|---|---|
| `train-images-idx3-ubyte.gz` | `8d4fb7e6c68d591d4c3dfef9ec88bf0d` |
| `train-labels-idx1-ubyte.gz` | `25c81989df183df01b3e8a0aad5dffbe` |
| `t10k-images-idx3-ubyte.gz` | `bef4ecab320f06d8554ea6380940ec79` |
| `t10k-labels-idx1-ubyte.gz` | `bb300cfdad3c16e7a12a480ee83cd310` |

## Not Reused

This dataset was not used in an earlier capstone project in this series.
