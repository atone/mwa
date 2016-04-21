package org.insightcentre.mono.aligners.jacana;


import java.io.File;
import java.io.IOException;
import java.util.List;

import org.insightcentre.mono.aligners.AlignmentTrainer;


public class JacanaAlignerTrainer implements AlignmentTrainer<JacanaAligner>{

	@Override
	public JacanaAligner train(
			List<AlignmentTrainer.TrainingPair> data) {
				return new JacanaAligner();
	}

	@Override
	public JacanaAligner load(File file) throws IOException {
		return new JacanaAligner();
	}

	public static void main(String[] args) {
		JacanaAlignerTrainer tra = new JacanaAlignerTrainer();
		tra.train(null);
	}

}
