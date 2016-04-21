package org.insightcentre.mono.aligners.jacana;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.insightcentre.mono.aligners.AlignmentTrainer;


public class JacanaAlignerTrainer1 implements AlignmentTrainer<JacanaAligner1>{

	@Override
	public JacanaAligner1 train(
			List<AlignmentTrainer.TrainingPair> data) {
				return new JacanaAligner1();
	}

	@Override
	public JacanaAligner1 load(File file) throws IOException {
		return new JacanaAligner1();
	}

	public static void main(String[] args) {
		JacanaAlignerTrainer1 tra = new JacanaAlignerTrainer1();
		tra.train(null);
	}

}
