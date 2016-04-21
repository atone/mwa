package org.insightcentre.mono.aligners;


import java.util.Arrays;

/**
 *
 * @author John McCrae
 */
public class Alignment {
    private final SentenceVectors sourceSentence, targetSentence;
    private final double[][] alignment;
    private String[] srcWrds;
    private String[] trgWrds;

    public Alignment(SentenceVectors sourceSentence, SentenceVectors targetSentence, double[][] alignment) {
        this.sourceSentence = sourceSentence;
        this.targetSentence = targetSentence;
        this.alignment = alignment;
        srcWrds = sourceSentence.words();
        trgWrds = targetSentence.words();
//        if(alignment.length != sourceSentence.size()) {
//            throw new IllegalArgumentException("source not size of matrix");
//        }
//        for (double[] alignment1 : alignment) {
//            if (alignment1.length != targetSentence.size()) {
//                throw new IllegalArgumentException("target not size of matrix");
//            }
//        }
    }

    public Alignment(Alignment alignment) {
        this.sourceSentence = alignment.sourceSentence;
        this.targetSentence = alignment.targetSentence;
        this.alignment = alignment.alignment;
        this.srcWrds = alignment.srcWrds;
        trgWrds = alignment.trgWrds;
    }
    
    public SentenceVectors getSourceSentence() {
        return sourceSentence;
    }
    
    public SentenceVectors getTargetSentence() {
        return targetSentence;
    }

    public int getSourceSize() {
        //return sourceSentence.size();
        return alignment.length;
    }

    public int getTargetSize() {
        //return targetSentence.size();
        return alignment[0].length;
    }
    
    public double alignment(int i, int j) {
        return alignment[i][j];
    }

    @Override
    public int hashCode() {
        int hash = 7;
        return hash;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final Alignment other = (Alignment) obj;
        return true;
    }

    @Override
    public String toString() {
        return "Alignment{\n" + "sourceSentence=" + sourceSentence + "\n targetSentence=" + targetSentence + "\n alignment=" + Arrays.deepToString(alignment) + "\n}";
    }

	public String[] getSrcWrds() {
		return srcWrds;
	}

	public void setSrcWrds(String[] srcWrds) {
		this.srcWrds = srcWrds;
	}

	public String[] getTrgWrds() {
		return trgWrds;
	}

	public void setTrgWrds(String[] trgWrds) {
		this.trgWrds = trgWrds;
	}
    
}
