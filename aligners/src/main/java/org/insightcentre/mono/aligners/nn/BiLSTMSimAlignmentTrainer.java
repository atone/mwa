package org.insightcentre.mono.aligners.nn;


import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.deeplearning4j.util.SerializationUtils;
import org.insightcentre.mono.aligners.AlignmentTrainer;

import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNImpl;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.af.Linear;
import edu.insight.unlp.nn.af.Sigmoid;
import edu.insight.unlp.nn.common.Sequence;
import edu.insight.unlp.nn.ef.SquareErrorFunction;
import edu.insight.unlp.nn.layers.FullyConnectedFFLayer;
import edu.insight.unlp.nn.layers.FullyConnectedLSTMLayer;

/**
 * Trainer for Bi-LSTM based aligner.
 * @author Kartik Asooja
 */
public class BiLSTMSimAlignmentTrainer implements AlignmentTrainer<BiLSTMSimAligner>{

	private SentenceDataset sentenceDataset = null;
	
	public void learnNN(NN nn, List<Sequence> training){	
		// size and no. of layers to be changed according to the word embedding used.
		NNLayer outputLayer = new FullyConnectedFFLayer(sentenceDataset.outputUnits, new Sigmoid(), nn);
		//NNLayer hiddenLayer2 = new FullyConnectedRNNLayer(sentenceDataset.outputUnits + 4, new Sigmoid(), nn);
		NNLayer hiddenLayer = new FullyConnectedLSTMLayer(sentenceDataset.inputUnits/2, new Sigmoid(), nn);
		NNLayer inputLayer = new FullyConnectedFFLayer(sentenceDataset.inputUnits, new Linear(), nn);
		List<NNLayer> layers = new ArrayList<NNLayer>();
		layers.add(inputLayer);
		layers.add(hiddenLayer);
		//layers.add(hiddenLayer2);
		layers.add(outputLayer);
		nn.setLayers(layers);
		nn.initializeNN();
		int epoch = 0;
		int maxEpochs = 10;
        //int maxEpochs = 10;
		while(epoch<maxEpochs) {
			epoch++;
			double trainingError = nn.sgdTrain(training, 0.0008, true);
			System.err.println("epoch["+epoch+"/" + maxEpochs + "] train loss = " + trainingError);
		}
	}

	@Override
	public BiLSTMSimAligner train(List<TrainingPair> data) {
		sentenceDataset = new SentenceDataset(data);
		NN fnn = new NNImpl(new SquareErrorFunction());
		NN bnn = new NNImpl(new SquareErrorFunction());
		learnNN(fnn, sentenceDataset.training);
		learnNN(bnn, sentenceDataset.backwardTraining);
		BiLSTMSimAligner biLSTMSimAligner = new BiLSTMSimAligner(fnn, bnn);
		return biLSTMSimAligner;
	}

	@Override
	public BiLSTMSimAligner load(File file) throws IOException {
		BiLSTMSimAligner biLSTMSimAligner = SerializationUtils.readObject(file);
		return biLSTMSimAligner;
	}
}

