import numpy
import numpy.linalg as la
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os, sys

DATASETS = os.path.join('.', 'data')

#Formula for calculating the grayscale component of pixels based on their rgb values
#https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
def rgb2gray(img):
    gray = numpy.zeros(img.shape)
    R = numpy.array(img[:, :, 0])
    G = numpy.array(img[:, :, 1])
    B = numpy.array(img[:, :, 2])

    R = (R *.299)
    G = (G *.587)
    B = (B *.114)

    Avg = (R+G+B)
    gray = img.copy()

    for i in range(3):
        gray[:,:,i] = Avg
    
    return gray

#format dict of similarities into a print-friendly string
def form_sim(sim):
    return os.linesep.join([f'{i[0]}: {i[1]:.04f}' for i in sorted(sim.items(), key=lambda i: i[1], reverse=True)])

#CONSTANTS
WIDTH = 175
HEIGHT = 200
MIDDLE = (WIDTH//2, HEIGHT//2)

NUM_SAMPLES = 10

#Show the dataset images as they are being loaded in, demonstration purposes
SHOW_DATASET_IMGS = False

#initialize zeroed face in shape (w*h,1)
avg = numpy.zeros(WIDTH*HEIGHT)

all = []  #final set of data, half first image examples, half second image examples

#load faces in form ./data/{dataset}/{dataset}nn.jpg and add to average
def load_face(dataset, show_images=True):
    global all, avg
    dataset_path = os.path.join(DATASETS,dataset)

    for i in range(NUM_SAMPLES):
        path = os.path.join(dataset_path, f'{dataset}{i:02}.jpg')
         
        #Check validity of filepath
        if not os.path.isfile(path):
            print(f"Invalid file: {path}")
            continue
        
        #Load the image at path into a matrix and display if chosen
        img = mpimg.imread(path)

        gray = rgb2gray(img)

        if show_images:
            imgplot = plt.imshow(gray)
            plt.show(block=False)
            plt.pause(0.2)
            plt.close()

        #Flatten grayscale image to 1D array and add to set
        r = numpy.reshape(gray[:,:,1], WIDTH*HEIGHT)
        all.append(r)

        avg = numpy.add(avg, r)

def get_average(datasets, show_images=True):
    global avg

    #load the faces of each dataset and add to avg
    for d in datasets:
        load_face(d, show_images=show_images)
    
    #Get the average of all images added to array and display
    avg = numpy.true_divide(avg, NUM_SAMPLES*len(datasets))

    if show_images:
        avgPixels = numpy.repeat(avg, 3)  #repeat out the grayscale value for each pixel rgb to show image

        avgPixels = numpy.reshape(avgPixels, (HEIGHT,WIDTH,3)).astype(int)

        plt.imshow(avgPixels)
        plt.show(block=False)
        plt.pause(1)
        plt.close()




def create_cloud(datasets, show_images=2):
    global all, avg
    get_average(datasets, show_images=SHOW_DATASET_IMGS)
    
    #Get "principal components" of each face by subtracting the cooresponding average value
    principal = numpy.zeros((WIDTH*HEIGHT,len(datasets)*NUM_SAMPLES))
    for j in range(len(datasets)*NUM_SAMPLES):
        principal[:,j] = all[j] - avg
        #fix shape and pixel values for image
        princImg = numpy.repeat(principal[:,j], 3)
        princImg = numpy.reshape(princImg, (HEIGHT,WIDTH,3)).astype(int)
    
    #Compute SVD
    (U,S,V) = la.svd(principal, full_matrices=False)
    phi = U[:,1:len(datasets)*NUM_SAMPLES]
    phi[:,1] = -1*phi[:,1]

    phiImg = numpy.reshape(phi,(HEIGHT*WIDTH,-1))   #group phi into set of H*W images

    if show_images:
        #display first 9 eigenfaces
        count = 1
        for i in range(3):
            for j in range(3):
                plt.subplot(3,3,count)
                im = numpy.repeat(phiImg[:, count], 3)
                im = numpy.reshape(im,(HEIGHT,WIDTH,3))
                plt.imshow(200-((25000*im).astype(int)))
                count+=1
                
        plt.show(block=False)
        plt.pause(show_images)
        plt.close()

    #Create clouds from dataset images after principal component analysis
    cloud1 = numpy.zeros(numpy.shape(principal[:,:NUM_SAMPLES]))
    for i in range(NUM_SAMPLES):
        imvec = principal[:,i]
        cloud1[:,i] = imvec.conj().T * phi[:,1] * phi[:,2] * phi[:,3]   #Transpose imvec onto first 3 svd
    
    cloud2 = numpy.zeros(numpy.shape(principal[:,:NUM_SAMPLES]))
    for i in range(NUM_SAMPLES):
        imvec = principal[:,NUM_SAMPLES+i]
        cloud2[:,i] = imvec.conj().T * phi[:,1] * phi[:,2] * phi[:,3]   #Transpose imvec onto first 3 svd

    if show_images:
        #Plot transposed images onto 3d figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(cloud1[1,:], cloud1[2,:], cloud1[3,:], c='r', label=datasets[0])
        ax.scatter(cloud2[1,:], cloud2[2,:], cloud2[3,:], c='b', label=datasets[1])
        ax.legend()
        plt.autoscale(enable=True, axis='both', tight=None)

        plt.show(block=False)
        plt.pause(show_images)
        plt.close()

    clouds = {datasets[0]:cloud1,datasets[1]:cloud2}
    return phi, clouds

    

def likeness(sample, phi, clouds, show_plot=2):
    global avg
    path = os.path.abspath(sample)
    keys = clouds.keys()

    #Check validity of filepath
    if not os.path.isfile(path):
        print(f"Invalid file: {path}")
        return
    
    #Load the image at path into a matrix and display for 1 second
    img = mpimg.imread(path)

    #grayscale loaded sample and subtract average of datasets
    gray = rgb2gray(img)
    u = numpy.reshape(gray[:,:,1], WIDTH*HEIGHT) - avg
    #transpose image data onto svd data
    upts = u.conj().T * phi[:,1] * phi[:,2] * phi[:,3]

    #print similarity of upts to each dataset
    sims = {}
    for k in keys:
        sims[k] = 0
        #for each point in cloud
        for i in range(len(clouds[k][1,:])):
            current = clouds[k][1:4,i]

            #cosine similarity between upts x,y,z axes and current point
            sims[k] += current.dot(upts[1:4]) / (numpy.linalg.norm(current) * numpy.linalg.norm(upts[1:4]))
        
        #average the similarities for this cloud, resulting in float from -1 to 1
        sims[k] /= len(clouds[k][1,:])
        sims[k] = (sims[k] + 1)/2   #normalize similarity value


    if show_plot:
        #Plot clouds
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        colors = ['r','b','g','c','m']
        for cloudi in clouds.items():
            cloud = cloudi[1]
            ax.scatter(cloud[1,:], cloud[2,:], cloud[3,:], c=colors.pop(0), label=cloudi[0])
        
        #Plot analyzed sample
        ax.scatter(upts[1], upts[2], upts[3], c='y', label=sample, marker='^', s=100)

        ax.legend()

        plt.show(block=False)
        plt.pause(show_plot)
        plt.close()
    
    return sims


def run_args(args):
    if '-h' in args or '--help' in args:
        print("Compare an input image to two training datasets to determine the similarity")
        print("Run with no arguments for default example")
        print("Usage:\tpython main.py sample_fp [options]*")

        print("Options:")
        print(f"\t{'-i':<10}Show eigenface images")
        print(f"\t{'-g':<10}Show graphs for the sample in relation to dataset images")
        print(f"\t{'--data':<10}Select which datasets to use\n\t{' ':10}Default: --data=jerma,arnold")
        print(f"\t{'--time':<10}How long to show images/graphs\n\t{' ':10}Default: --time=3")
    
        sys.exit(0)

    sample = args.pop(0)

    #defaults
    show_images = False
    show_graphs = False
    datasets = ['jerma', 'arnold']
    time = 3

    #parse
    for arg in args:

        if arg.startswith('--'):
            keyname = arg[2:arg.index('=')]

            if keyname == 'data':
                #split passed datasets on ,
                values = arg[arg.index('=')+1:].split(',')
                datasets = values
            elif keyname == 'time':
                value = arg[arg.index('=')+1:]
                try:
                    time = int(value)
                except ValueError:
                    print("Invalid time value, using default")
            
            continue

        if arg.startswith('-'):
            #parse simple option
            for o in arg[1:]:
                if o == 'i':
                    show_images = True
                elif o == 'g':
                    show_graphs = True
            
            continue

    #convert show_ variables to None or length
    show_images = time if show_images else None
    show_graphs = time if show_graphs else None

    #analyze the datasets and create point clouds
    phi, clouds = create_cloud(datasets, show_images=show_images)

    #compare sample image to the clouds
    sim = likeness(sample, phi, clouds, show_plot=show_graphs)
    print(f"I think this image is of {max(sim, key=sim.get)}!")
    print(f"Similarities: \n{form_sim(sim)}")      
            

if __name__ == "__main__":
    print("Eigen Action Heros")

    if len(sys.argv) > 1:
        run_args(sys.argv[1:])
        sys.exit(0)

    phi, clouds = create_cloud(["jerma", "arnold"], show_images=None)

    sim = likeness("test06.jpg", phi, clouds, show_plot=2)
    print(f"I think this image is of {max(sim, key=sim.get)}!")
    print(f"Similarities: \n{form_sim(sim)}")  
    
    sim = likeness("test00.jpg", phi, clouds, show_plot=None)
    print(f"I think this image is of {max(sim, key=sim.get)}!")
    print(f"Similarities: \n{form_sim(sim)}")  
