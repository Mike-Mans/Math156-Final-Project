# Speech Command Classification Using Shallow Convolutional Neural Networks on Mel Spectrograms

Michael Mansour, Ezekiel Ito, Gorm Kragh, Adrian Ruiz Doblas, Jadon Zhu

## Abstract
We present a shallow convolutional neural network (CNN) designed and implemented from scratch to classify one-second audio recordings of spoken commands. Our approach transforms raw audio waveforms into Mel spectrograms using the Short-Time Fourier Transform (STFT), providing 2D time-frequency representations suitable for CNN processing. The model is trained on the Google Speech Commands dataset to distinguish between 10 core command words (yes, no, up, down, left, right, on, off, stop, go), along with an ``unknown'' class for out-of-vocabulary words and a ``silence'' class. The architecture consists of two convolutional blocks with batch normalization and max pooling, followed by adaptive pooling and fully connected layers producing 12-class predictions. We employ automated hyperparameter optimization using Optuna, extensive data augmentation (time shifting, noise addition, and spectrogram masking), and Automatic Mixed Precision training. The final model achieves 81\% test accuracy with weighted average precision and recall of 0.81. The results demonstrate that a relatively simple CNN architecture can achieve reasonable performance on speech command classification when combined with appropriate pre-processing and training techniques.
\end{abstract}

## Introduction
Speech command classification is a fundamental task in keyword spotting systems (KWS), which enable voice-activated devices to recognize specific spoken commands from continuous audio streams. The challenge lies in accurately identifying short, isolated command words (typically one second or less) while distinguishing them from background noise, out-of-vocabulary words, and silence.

Our project aims to design, implement, and train a shallow convolutional neural network (CNN) from scratch to classify one-second audio recordings of spoken commands into simple words. We transform 1D audio waveforms into 2D Mel spectrograms using the Short-Time Fourier Transform (STFT), providing time-frequency representations that capture the acoustic characteristics of speech. The model distinguishes between 10 core command words (yes, no, up, down, left, right, on, off, stop, go), along with an ``unknown'' class for out-of-vocabulary words and a ``silence'' class for background noise.

CNNs are special types of neural networks that are better fitted for spatial data. As discussed in O'Shea and Nash's paper, CNNs are more suitable for image-focused tasks compared to typical neural networks as the convolutional layer focuses on learnable kernels.~\cite{OSheaNash2015} CNNs are particularly well-suited for this task due to several key advantages. First, spectrograms are inherently 2D spatial representations with time along one axis and frequency along the other, making them natural inputs for convolutional operations. Second, CNNs involve far fewer parameters than fully connected networks by sharing weights across spatial locations, making them more resistant to overfitting and computationally efficient.

## Background

In this section we briefly review supervised classification with neural networks and the basic ideas behind convolutional neural networks (CNNs). These concepts provide the theoretical foundation for the architecture and training procedure used in our project.

### Supervised classification with neural networks

We consider a labeled dataset $\mathcal{D} = \{(x_n, t_n)\}_{n=1}^N$, where each input $x_n$ is an instance and $t_n \in \{e_1,\dots,e_K\} \subset \mathbb{R}^K$ is a one–hot label for $K$ classes. A neural network defines a parametric mapping $f_\theta : \mathcal{X} \to \Delta^{K-1}$ from inputs to class probabilities, where $\theta$ collects all weights and biases and $\Delta^{K-1}$ denotes the probability simplex (Bishop, 2006, Sec.~4.3).

In a standard feed–forward network, layers are defined recursively by $h^{(0)} = x$, $a^{(\ell)} = W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}$, and $h^{(\ell)} = \phi(a^{(\ell)})$, where $\phi$ is a nonlinear activation function (Bishop, 2006, Sec.~5.1). For multiclass classification, the final layer uses a softmax activation, producing class probabilities $y_k(x) = \exp(a^{(L)}_k) / \sum_{j=1}^K \exp(a^{(L)}_j)$ (Bishop, 2006, Sec.~4.3.2). Training is performed by minimizing the cross–entropy loss 

\begin{equation*}
    \mathcal{L}(\theta) = - \sum_{n=1}^N \sum_{k=1}^K t_{nk} \log y_k(x_n),
\end{equation*}
with gradients computed via backpropagation (Bishop, 2006, Secs.~4.3, 5.2.4, 5.3).

### Convolutional neural networks

Fully connected networks ignore the spatial structure of their inputs by flattening them into vectors. CNNs instead exploit locality and approximate translation invariance by using local receptive fields, weight sharing, and pooling (Bishop, 2006, Sec.~5.5.6).

For an input $x \in \mathbb{R}^{H \times W \times C_{\mathrm{in}}}$, a convolutional filter $W^{(k)} \in \mathbb{R}^{F_h \times F_w \times C_{\mathrm{in}}}$ computes the discrete convolution $(x * W^{(k)})_{i,j} = \sum_{u,v,c} W^{(k)}_{u,v,c} \, x_{i+u,\,j+v,\,c}$ at each spatial location $(i,j)$, producing feature maps $h^{(k)}_{i,j} = \phi((x * W^{(k)})_{i,j} + b^{(k)})$ where $\phi$ is an activation function. We use the rectified linear unit (ReLU), $\phi(a)=\max\{0,a\}$, which is standard in modern CNNs. Stacking multiple filters yields output tensors whose dimensions depend on filter size, stride, and padding. After convolutional and pooling layers, the output is flattened and fed to fully connected layers followed by a softmax layer.

Pooling layers aggregate activations in local neighborhoods to build coarser, more translation-invariant feature maps. Max-pooling over $2\times 2$ windows keeps the maximum activation in each window, while average pooling keeps the mean (Bishop, 2006, Sec.~5.5.6). The entire network is trained end-to-end by backpropagation using the cross-entropy loss.


## Dataset

### Dataset Description and Source

We use the Google Speech Commands dataset v0.02 \cite{speechcommands_hf, warden2018speechcommands}, which contains 105,829 one-second audio recordings as .wav files sampled at 16 kHz. The dataset was collected through crowdsourcing and contains recordings from thousands of different speakers, with each file clearly labeled by its spoken word class. This labeling serves as our target value for supervised learning. For related CNN approaches on this dataset, see \cite{tang2018deepresidual, majumdar2020matchboxnet}.

We organize the dataset into 12 classes for classification. The 10 core command words (yes, no, up, down, left, right, on, off, stop, go) serve as distinct classes. The remaining 20 auxiliary words (zero, one, two, three, four, five, six, seven, eight, nine, bed, bird, cat, dog, happy, house, Marvin, Sheila, tree, wow) are grouped into a single ``unknown'' class, requiring the model to identify them as out-of-vocabulary words rather than distinguish between them. Finally, we create a ``silence'' class by extracting one-second chunks from the background noise files provided in the dataset, ensuring they match the duration of other audio samples.

### Dataset Size and Class Distribution

The dataset is pre-split into training, validation, and test sets by the dataset creators, with mutually exclusive speakers across splits to ensure proper generalization evaluation. We use these predefined splits without modification. The test set contains 11,046 samples with the following class distribution: the ``unknown'' class dominates with 6,931 samples (62.8\%), each of the 10 core command classes contains approximately 400 samples (ranging from 396 to 425 samples, approximately 3.6\% each), and the ``silence'' class contains 41 samples (0.4\%). This significant class imbalance poses challenges for model training, as the model may bias predictions toward the dominant ``unknown'' class.

### Ethical Concerns

The Google Speech Commands dataset was collected through crowdsourcing, which raises several ethical considerations. First, while the dataset includes diverse speakers, the demographic distribution may not be fully representative of global populations, potentially leading to performance disparities across different demographic groups. Second, the dataset contains recordings of individuals' voices, raising privacy concerns; however, the dataset is released under the Creative Commons BY 4.0 license with appropriate consent mechanisms. Third, speech recognition systems trained on such datasets may perpetuate biases if certain accents, dialects, or speech patterns are underrepresented. We acknowledge these concerns and note that our model is intended for research and educational purposes, with careful consideration needed before deployment in real-world applications.

### Pre-processing

We convert raw audio waveforms into 2D Mel spectrograms suitable for CNN processing. Audio files are first converted from stereo to mono (if necessary) and adjusted to exactly one second duration (16,000 samples at 16 kHz), with shorter files padded with silence and longer files truncated. We then use PyTorch's Torchaudio library to generate Mel spectrograms using \texttt{torchaudio.transforms.MelSpectrogram} with 128 mel bins, which provide a frequency representation aligned with human auditory perception by mapping frequencies to the Mel scale and emphasizing perceptually relevant frequency ranges. The transformation uses a Short-Time Fourier Transform (STFT) with appropriate windowing and hop length parameters. Finally, spectrograms are converted to logarithmic scale (log-magnitude) to compress the dynamic range and improve numerical stability during training.

### Data Augmentation

To improve model generalization and robustness, we apply several data augmentation techniques during training (augmentations are disabled for validation and test sets). Waveforms are randomly shifted left or right by up to 0.2 seconds to simulate variations in utterance timing. Random snippets from background noise files are mixed with audio samples at random Signal-to-Noise Ratios (SNR) between 5 and 20 dB to improve robustness to environmental noise. Random horizontal bands are masked in the spectrogram (masking parameter tuned via Optuna, range 10--40 time frames) to simulate temporal occlusions, while random vertical bands are masked (masking parameter tuned via Optuna, range 5--20 frequency bins) to simulate frequency occlusions and improve robustness to frequency variations. These augmentation techniques are applied stochastically during training, effectively increasing the diversity of training examples and reducing overfitting. The augmentation parameters (time and frequency masking ranges) were optimized as part of the hyperparameter search process described in Section~\ref{sec:model}.

### Exploratory Data Analysis

During our exploratory data analysis, we generated figures such as Figure~\ref{fig:spectrogram-example}, which shows an example Mel spectrogram generated from a training sample, illustrating the time-frequency representation used as input to our CNN.

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{data/spectrogram.png}
\caption{Example Mel spectrogram visualization from the Google Speech Commands dataset. The horizontal axis represents time, and the vertical axis represents frequency (Mel scale). Brighter regions indicate higher energy at specific time-frequency locations, revealing patterns characteristic of speech.}
\label{fig:spectrogram-example}
\end{figure} 

## Model

We implemented a shallow convolutional neural network from scratch specifically designed for speech command classification. The architecture, training procedure, and evaluation methodology are described below.

### Architecture

Our ShallowSpeechCNN model is a custom architecture implemented from scratch using PyTorch. The network consists of two convolutional blocks followed by fully connected layers:

\begin{itemize}
    \item \textbf{Convolutional Block 1}: 2D convolution (16 channels, $3 \times 3$ kernel), batch normalization, ReLU activation, and max pooling ($2 \times 2$).
    
    \item \textbf{Convolutional Block 2}: 2D convolution (32 channels, $3 \times 3$ kernel), batch normalization, ReLU activation, and max pooling ($2 \times 2$).
    
    \item \textbf{Adaptive Pooling}: Adaptive average pooling to a fixed $4 \times 4$ feature map.
    
    \item \textbf{Fully Connected Layers}: Two linear layers mapping from $32 \times 4 \times 4 = 512$ features to 64 hidden units (ReLU), then to 12 output classes (10 core commands, plus ``unknown'' and ``silence'').
\end{itemize}

Batch normalization stabilizes training, and adaptive pooling allows handling spectrograms of varying temporal lengths.

### Baseline Implementation and Improvements

Our initial baseline implementation used a standard linear spectrogram transform (\texttt{torchaudio.\allowbreak transforms.\allowbreak Spectrogram}) with instance normalization applied to log-scaled spectrograms. The CNN architecture itself remained unchanged between baseline and improved versions. The baseline employed minimal data augmentation (only random time shifting, as described in Section~\ref{sec:dataset}) and hardcoded hyperparameters (batch size 64, Adam optimizer with learning rate 0.001, 10 training epochs).

The improved implementation introduced several key changes: (1) replacement of linear spectrograms with Mel spectrograms (see Section~\ref{sec:dataset} for pre-processing details); (2) comprehensive data augmentation (see Section~\ref{sec:dataset}); (3) automated hyperparameter optimization using Optuna; and (4) training efficiency improvements including Automatic Mixed Precision (AMP) and enhanced checkpointing for resumable training. These modifications improved feature representation and training resilience while maintaining the same shallow CNN architecture.

### Training Procedure

Training was performed with hyperparameter optimization and several efficiency improvements.

### Loss Function and Optimization

We used the cross-entropy loss function (as defined in Section~\ref{sec:background}) for multiclass classification. Hyperparameters (optimizer, learning rate, and batch size) were optimized using Optuna~\cite{optuna2019} through automated search on a validation subset, selecting optimal values from candidate ranges for Adam/RMSprop/SGD optimizers, learning rates in $[10^{-5}, 10^{-2}]$, and batch sizes in $\{32, 64, 128\}$. The validation set was used exclusively for hyperparameter selection, ensuring no information leakage to the test set.

\subsubsection{Training Configuration}

The final model was trained for 20 epochs using the best hyperparameters identified by Optuna. We implemented Automatic Mixed Precision (AMP) to improve training efficiency and checkpointing to save model state after each epoch if validation loss improves. Training was performed with reproducibility measures including fixed random seeds.


### Evaluation Metrics

Model performance was evaluated using cross-entropy loss, classification accuracy, confusion matrices, and per-class precision, recall, and F1-scores. The test set was used \emph{only} for final evaluation after all hyperparameter tuning and model selection were complete, ensuring no data leakage from the test set into the training process. 

## Results

We present the performance results of our trained CNN model on the test set, along with a comparison to our initial baseline implementation. The final model achieved an overall test accuracy of 81\%.

### Overall Performance

The model achieved a weighted average precision of 0.81, recall of 0.81, and F1-score of 0.79 across all 12 classes on the test set (11,046 samples). The macro-averaged metrics (precision: 0.78, recall: 0.59, F1-score: 0.66) reflect the class imbalance in the dataset, where the ``unknown'' class dominates with 6,931 samples compared to the core command classes (approximately 400 samples each) and the ``silence'' class (41 samples).

### Training Dynamics

Figure~\ref{fig:training-curves} shows the training and validation loss, along with validation accuracy, over the 20 training epochs. Both training and validation loss decrease throughout training, indicating that the model continued to learn and improve without signs of overfitting. The validation accuracy similarly increases steadily, reaching its peak at the final epoch. This suggests that training for the full 20 epochs was beneficial and that the model had not yet reached its full learning capacity. The training loss appears smoother than the validation loss due to batch averaging and data augmentation.

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{data/training_validation_curves.png}
\caption{Training and validation loss, and validation accuracy over 20 epochs. Both loss curves decrease, and validation accuracy increases steadily throughout training.}
\label{fig:training-curves}
\end{figure}

## Per-Class Performance

Table~\ref{tab:per-class-metrics} presents precision, recall, and F1-scores for each class. The model demonstrates strong performance on several core commands: ``yes'' achieves the highest F1-score among core commands (0.85), and ``stop'' performs well (F1-score 0.76). The ``unknown'' class achieves excellent performance (F1-score 0.88, recall 0.96), indicating successful identification of out-of-vocabulary words.

\begin{table}[H]
\centering
\caption{Per-class precision, recall, F1-score, and support on the test set.}
\label{tab:per-class-metrics}
\begin{tabular}{lcccc}
\toprule
\textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Support} \\
\midrule
yes & 0.83 & 0.88 & 0.85 & 419 \\
no & 0.75 & 0.39 & 0.51 & 405 \\
up & 0.83 & 0.49 & 0.62 & 425 \\
down & 0.66 & 0.64 & 0.65 & 406 \\
left & 0.84 & 0.44 & 0.58 & 412 \\
right & 0.86 & 0.58 & 0.69 & 396 \\
on & 0.76 & 0.55 & 0.64 & 396 \\
off & 0.80 & 0.52 & 0.63 & 402 \\
stop & 0.79 & 0.73 & 0.76 & 411 \\
go & 0.67 & 0.36 & 0.47 & 402 \\
unknown & 0.82 & 0.96 & 0.88 & 6931 \\
silence & 0.71 & 0.59 & 0.64 & 41 \\
\bottomrule
\end{tabular}
\end{table}

Some classes exhibit lower recall: ``go'' (0.36), ``no'' (0.39), and ``up'' (0.49), which may reflect acoustic similarities between certain command pairs or insufficient training examples. The ``silence'' class achieves moderate performance (F1-score 0.64), reasonable given its small sample size (41 test samples).

### Confusion Matrix Analysis

The confusion matrix (Table~\ref{tab:confusion-matrix}) reveals several notable misclassification patterns.

\begin{table}[H]
\centering
\caption{Confusion matrix showing predicted vs.\ actual class labels on the test set. Rows represent true labels, columns represent predictions. Diagonal entries indicate correct classifications.}
\label{tab:confusion-matrix}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccccccc}
\toprule
 & \textbf{yes} & \textbf{no} & \textbf{up} & \textbf{down} & \textbf{left} & \textbf{right} & \textbf{on} & \textbf{off} & \textbf{stop} & \textbf{go} & \textbf{unknown} & \textbf{silence} \\
\midrule
\textbf{yes} & 367 & 0 & 0 & 0 & 3 & 0 & 0 & 0 & 0 & 0 & 49 & 0 \\
\textbf{no} & 1 & 158 & 1 & 47 & 1 & 0 & 0 & 0 & 2 & 23 & 172 & 0 \\
\textbf{up} & 0 & 0 & 209 & 0 & 1 & 0 & 8 & 33 & 31 & 0 & 142 & 1 \\
\textbf{down} & 0 & 5 & 1 & 258 & 0 & 0 & 3 & 0 & 1 & 3 & 135 & 0 \\
\textbf{left} & 54 & 0 & 1 & 0 & 183 & 7 & 0 & 0 & 1 & 0 & 166 & 0 \\
\textbf{right} & 0 & 0 & 1 & 0 & 9 & 228 & 0 & 0 & 0 & 0 & 157 & 1 \\
\textbf{on} & 0 & 0 & 2 & 1 & 0 & 0 & 217 & 8 & 0 & 0 & 167 & 1 \\
\textbf{off} & 1 & 0 & 25 & 0 & 3 & 1 & 11 & 211 & 2 & 0 & 148 & 0 \\
\textbf{stop} & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 298 & 0 & 112 & 0 \\
\textbf{go} & 0 & 34 & 0 & 29 & 1 & 0 & 1 & 1 & 4 & 143 & 188 & 1 \\
\textbf{unknown} & 20 & 13 & 11 & 54 & 16 & 28 & 47 & 12 & 38 & 44 & 6642 & 6 \\
\textbf{silence} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 17 & 24 \\
\bottomrule
\end{tabular}%
}
\end{table}

Notable misclassifications include ``go''/``no'' confusion (34 and 23 samples), ``up''/``off'' confusion (33 and 25 samples), and frequent misclassification of core commands as ``unknown''. This reflects both acoustic similarities between certain pairs and the class imbalance favoring the ``unknown'' class.

### Comparison with Baseline

Compared to our baseline, the improved model achieved gains across classes through Mel spectrograms, comprehensive data augmentation, and automated hyperparameter optimization. Some notable improvements: ``no'' (F1: 0.28 to 0.51), ``on'' (F1: 0.13 to 0.64), ``down'' (F1: 0.52 to 0.65), and ``go'' (F1: 0.41 to 0.47).

## Conclusions and Discussion

Our results demonstrate that shallow CNNs can achieve reasonable performance (81\% accuracy) on speech command classification when combined with appropriate pre-processing and training techniques. The model generalizes well to out-of-vocabulary words, suggesting that the learned features capture meaningful acoustic patterns beyond the specific training vocabulary.

However, several limitations emerged from our analysis. The confusion matrix reveals systematic misclassifications between phonetically similar pairs (e.g., ``go''/``no'', ``up''/``off''), suggesting that the shallow architecture may lack the representational capacity to distinguish subtle acoustic differences. Additionally, the model's tendency to misclassify core commands as ``unknown'' reflects the class imbalance in the dataset, with minority classes like ``go'' and ``no'' achieving low recall despite reasonable precision.

Future work could address these limitations through deeper architectures with residual connections or attention mechanisms, class balancing techniques such as weighted loss functions or focal loss, transfer learning from larger speech recognition models, or ensemble methods.

\newpage

## Author contributions

Jadon Zhu wrote the hyperparameter tuning logic using Optuna and helped architect the shallow CNN.

Ezekiel Ito wrote the first draft of the pre-processing code allowing us to better visualize the data and download individual spectrograms. 

Adrián Ruiz Doblas and Michael Mansour handled the pre-processing, including converting audio files to usable spectrograms via Short-Time Fourier Transform, and also helped debug various other parts of the code.

Gorm Kragh contributed code review, proofreading, and ideas on data pre-processing.

Every one of us contributed to the written report.

## Acknowledgments

The model inspiration came from the popular ways to analyze the famous Google Speech Commands dataset.

