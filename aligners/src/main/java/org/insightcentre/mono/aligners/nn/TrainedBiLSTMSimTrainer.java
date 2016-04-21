package org.insightcentre.mono.aligners.nn;


import java.io.File;
import java.io.IOException;
import java.util.List;

import org.deeplearning4j.util.SerializationUtils;
import org.insightcentre.mono.aligners.AlignmentTrainer;

/**
 * Trainer for Bi-LSTM based aligner.
 * @author Kartik Asooja
 */
public class TrainedBiLSTMSimTrainer implements AlignmentTrainer<BiLSTMSimAligner>{

	private final static String SCORE_THRESHOLD_STRING = "05_25000";

	private String toSerializeModelPath = "models/" + "bilstmComposes_" + SCORE_THRESHOLD_STRING + ".model";
	private static BiLSTMSimAligner bilstmSimAligner = null;

	private File bilstmModel = new File(toSerializeModelPath);

	public TrainedBiLSTMSimTrainer() {
		try {
			bilstmSimAligner = load(bilstmModel);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public BiLSTMSimAligner train(List<TrainingPair> data) {
		return bilstmSimAligner;
	}

	@Override
	public BiLSTMSimAligner load(File file) throws IOException {
		BiLSTMSimAligner bilstmSimAligner = SerializationUtils.readObject(file);
		return bilstmSimAligner;
	}
}

