package org.insightcentre.mono.aligners.nn;


import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.util.SerializationUtils;
import org.insightcentre.mono.aligners.Aligner;
import org.insightcentre.mono.aligners.AlignmentTrainer;

import edu.insight.unlp.nn.DataSet;
import edu.insight.unlp.nn.NANN;
import edu.insight.unlp.nn.NANNImpl;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.af.Linear;
import edu.insight.unlp.nn.af.Sigmoid;
import edu.insight.unlp.nn.common.NASequence;
import edu.insight.unlp.nn.ef.SquareErrorFunction;
import edu.insight.unlp.nn.layers.FullyConnectedFFLayer;
import edu.insight.unlp.nn.layers.NAFCRNNLayer;

public class NABiLSTMAlignmentTrainer implements AlignmentTrainer<NABiLSTMAligner> {

	private DataSet dataset;
	private BiLSTMSimAligner aligner;

	public NABiLSTMAlignmentTrainer(Aligner aligner) {
		this.aligner = (BiLSTMSimAligner) aligner;
	}

	@Override
	public NABiLSTMAligner train(List<TrainingPair> data) {
		dataset = new SentencePairBiRNNDataset(data, aligner);
		NASequence naSequence = (NASequence) dataset.training.get(0);

		int fRNNTargetVectorSize = naSequence.naInput[0].length;
		int annotationVectorHjSize = naSequence.inputSeq[0].length;

		NANN naNN = new NANNImpl(new SquareErrorFunction());
		NNLayer naInputLayer = new FullyConnectedFFLayer(fRNNTargetVectorSize + annotationVectorHjSize, new Linear(), naNN);
		NNLayer naHiddenLayer = new FullyConnectedFFLayer(40, new Sigmoid(), naNN); 
		NNLayer naOutputLayer = new FullyConnectedFFLayer(1, new Sigmoid(), naNN); //output is just the attention weight
		List<NNLayer> naLayers = new ArrayList<NNLayer>();
		naLayers.add(naInputLayer);
		naLayers.add(naHiddenLayer);
		naLayers.add(naOutputLayer);
		naNN.setLayers(naLayers);
		naNN.initializeNN();

		NN nn = new NANNImpl(new SquareErrorFunction(), naNN);		
		NNLayer outputLayer = new FullyConnectedFFLayer(dataset.outputUnits, new Sigmoid(), nn);		
		NNLayer hiddenLayer1 = new NAFCRNNLayer(dataset.outputUnits/4, new Sigmoid(), nn);
		NNLayer hiddenLayer2 = new NAFCRNNLayer(annotationVectorHjSize/8, new Sigmoid(), nn);
		NNLayer inputLayer = new FullyConnectedFFLayer(annotationVectorHjSize, new Linear(), nn);

		List<NNLayer> layers = new ArrayList<NNLayer>();
		layers.add(inputLayer);
		layers.add(hiddenLayer1);
		layers.add(hiddenLayer2);
		layers.add(outputLayer);
		nn.setLayers(layers);
		nn.initializeNN();

		int epoch = 0;
		int maxEpochs = 8;
		while(epoch<maxEpochs) {
			epoch++;
			double trainingError = nn.sgdTrain(dataset.training, 0.0004, true);
			System.out.println("epoch["+epoch+"/" + maxEpochs + "] train loss = " + trainingError);
		}

		NABiLSTMAligner naAligner = new NABiLSTMAligner(naNN, aligner);
		return naAligner;
	}

	@Override
	public NABiLSTMAligner load(File file) throws IOException {
		NABiLSTMAligner naAligner = SerializationUtils.readObject(file);
		return naAligner;
	}

}
