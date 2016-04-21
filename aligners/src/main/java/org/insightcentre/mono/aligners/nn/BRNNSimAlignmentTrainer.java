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
import edu.insight.unlp.nn.layers.FullyConnectedRNNLayer;

/**
 * Trainer for Bi-RNN based aligner.
 * @author Kartik Asooja
 */
public class BRNNSimAlignmentTrainer implements AlignmentTrainer<BRNNSimAligner>{

	private SentenceDataset sentenceDataset = null;
	
	public void learnNN(NN nn, List<Sequence> training){	
		// size and no. of layers to be changed according to the word embedding used.
		NNLayer outputLayer = new FullyConnectedFFLayer(sentenceDataset.outputUnits, new Sigmoid(), nn);
		//NNLayer hiddenLayer2 = new FullyConnectedRNNLayer(sentenceDataset.outputUnits + 4, new Sigmoid(), nn);
		NNLayer hiddenLayer = new FullyConnectedRNNLayer(sentenceDataset.inputUnits/2, new Sigmoid(), nn);
		NNLayer inputLayer = new FullyConnectedFFLayer(sentenceDataset.inputUnits, new Linear(), nn);
		List<NNLayer> layers = new ArrayList<NNLayer>();
		layers.add(inputLayer);
		layers.add(hiddenLayer);
		//layers.add(hiddenLayer2);
		layers.add(outputLayer);
		nn.setLayers(layers);
		nn.initializeNN();
		int epoch = 0;
		int maxEpochs = 40;
        //int maxEpochs = 10;
		while(epoch<maxEpochs) {
			epoch++;
			double trainingError = nn.sgdTrain(training, 0.0008, true);
			System.err.println("epoch["+epoch+"/" + maxEpochs + "] train loss = " + trainingError);
		}
	}

	@Override
	public BRNNSimAligner train(List<TrainingPair> data) {
		sentenceDataset = new SentenceDataset(data);
		NN fnn = new NNImpl(new SquareErrorFunction());
		NN bnn = new NNImpl(new SquareErrorFunction());
		learnNN(fnn, sentenceDataset.training);
		learnNN(bnn, sentenceDataset.backwardTraining);
		BRNNSimAligner brnnSimAligner = new BRNNSimAligner(fnn, bnn);
		return brnnSimAligner;
	}

	@Override
	public BRNNSimAligner load(File file) throws IOException {
		BRNNSimAligner brnnSimAligner = SerializationUtils.readObject(file);
		return brnnSimAligner;
	}
}

