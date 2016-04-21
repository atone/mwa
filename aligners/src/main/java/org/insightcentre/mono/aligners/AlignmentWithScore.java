package org.insightcentre.mono.aligners;


/**
 * An alignment with a score
 * @author John McCrae
 */
public class AlignmentWithScore extends Alignment {
    public final double score;

    public AlignmentWithScore(double score, SentenceVectors sourceSentence, SentenceVectors targetSentence, double[][] alignment) {
        super(sourceSentence, targetSentence, alignment);
        this.score = score;
    }

    public AlignmentWithScore(Alignment align, double score) {
        super(align);
        this.score = score;
    }
    
    @Override
    public int hashCode() {
        int hash = 3;
        hash = 31 * hash + (int) (Double.doubleToLongBits(this.score) ^ (Double.doubleToLongBits(this.score) >>> 32));
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
        final AlignmentWithScore other = (AlignmentWithScore) obj;
        if (Double.doubleToLongBits(this.score) != Double.doubleToLongBits(other.score)) {
            return false;
        }
        return true;
    }


}
