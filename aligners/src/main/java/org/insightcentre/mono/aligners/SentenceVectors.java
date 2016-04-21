package org.insightcentre.mono.aligners;


import java.util.Arrays;

/**
 *
 * @author John McCrae
 */
public class SentenceVectors {
    public final double[][] vectors;
    private final String[] words;
    private final String original;

    public SentenceVectors(double[][] vectors, String[] words, String original) {
        if(words.length == 0) {
            throw new IllegalArgumentException("Empty sentence");
        }
        this.vectors = vectors;
        this.words = words;
        this.original = original;
        for(int i = 1; i < vectors.length; i++) {
            if(vectors[i].length != vectors[0].length) {
                throw new IllegalArgumentException("Vectors not all of same length");
            }
        }
        if(words.length != vectors.length) {
            throw new IllegalArgumentException("Vectors and words do not have same alignment");
        }
    }
    
    public final String word(int i) {
        return words[i];
    }

    public final String original() {
        return original;
    }
    
    public final double[] vector(int i) {
        return vectors[i];
    }

    public final String[] words() {
        return words;
    }
    
    public final int size() {
        return vectors.length;
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 37 * hash + Arrays.deepHashCode(this.vectors);
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
        final SentenceVectors other = (SentenceVectors) obj;
        if (!Arrays.deepEquals(this.vectors, other.vectors)) {
            return false;
        }
        return true;
    }
    
    @Override
    public String toString() {
        final StringBuffer sb = new StringBuffer();
        for(int i = 0; i < size(); i++) {
            if(i > 0)
                sb.append(" ");
            sb.append(words[i]);
            sb.append(" [");
            for(int j = 0; j < Math.min(vectors[i].length, 6); j++) {
                sb.append(String.format("%.3f", vectors[i][j]));
                sb.append(" ");
            }
            if(vectors[i].length > 6) 
                sb.append("...");
            sb.append("]");
        }
        return sb.toString();
    }
    
}
