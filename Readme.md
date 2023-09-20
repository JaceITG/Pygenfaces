# Pygenfaces

Pygenfaces is a Python adaptation of Steve Brunton's [SVD Eigen Action Hero Project](https://www.youtube.com/watch?v=SD4NfEKZ_p8), which tests an input image for its similarity to provided datasets of faces.

## Usage

```
> python3 main.py sample_fp [options, refer to -h]*
```

## Example

Analyzing the similarity of test image of Arnold

![arnold](https://github.com/JaceITG/Pygenfaces/blob/main/test00.jpg)


Run Pygenfaces on the 'jerma' and 'arnold' datasets

```
> python3 main.py test00.jpg -g --data=jerma,arnold
```

After processing all dataset images, the following Eigenfaces are created

![eigenfaces](https://github.com/JaceITG/Pygenfaces/blob/main/example/eigenexample.jpg)

The following graph depicts the eigenspace values of each dataset's images, along with the placement of the sample image

![graph](https://github.com/JaceITG/Pygenfaces/blob/main/example/example.jpg)

Output showing quantitative similarity of sample to each dataset:
```
I think this image is of arnold!
Similarities:
arnold: 0.5262584242768676
jerma: 0.4111224493153606
```
