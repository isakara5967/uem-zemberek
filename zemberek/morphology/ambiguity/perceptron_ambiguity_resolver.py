from __future__ import annotations

from typing import List, TYPE_CHECKING, DefaultDict, Any, Optional, Tuple, Dict

from operator import attrgetter
from collections import defaultdict, OrderedDict
import numpy as np

if TYPE_CHECKING:
    from zemberek.core.data.weight_lookup import WeightLookup

from zemberek.core.data.compressed_weights import CompressedWeights
from zemberek.core.turkish.secondary_pos import SecondaryPos
from zemberek.morphology.ambiguity.ambiguity_resolver import AmbiguityResolver
from zemberek.morphology.analysis.sentence_analysis import SentenceAnalysis
from zemberek.morphology.analysis.word_analysis import WordAnalysis
from zemberek.morphology.analysis.single_analysis import SingleAnalysis
from zemberek.morphology.analysis.sentence_word_analysis import SentenceWordAnalysis


class PerceptronAmbiguityResolver(AmbiguityResolver):
    sentence_begin: SingleAnalysis = SingleAnalysis.unknown("<s>")
    sentence_end: SingleAnalysis = SingleAnalysis.unknown("</s>")

    def __init__(self, averaged_model: WeightLookup, extractor: 'PerceptronAmbiguityResolver.FeatureExtractor'):
        self.decoder = PerceptronAmbiguityResolver.Decoder(averaged_model, extractor)

    @classmethod
    def from_resource(cls, resource_path: str) -> 'PerceptronAmbiguityResolver':
        lookup = CompressedWeights.deserialize(resource_path)
        extractor = cls.FeatureExtractor(use_cache=True)
        return cls(lookup, extractor)

    def disambiguate(self, sentence: str, all_analyses: List[WordAnalysis]) -> SentenceAnalysis:
        best: PerceptronAmbiguityResolver.DecodeResult = self.decoder.best_path(all_analyses)

        l: List[SentenceWordAnalysis] = [
            SentenceWordAnalysis(best.best_parse[i], word_analysis) for i, word_analysis in enumerate(all_analyses)
        ]
        return SentenceAnalysis(sentence, l)

    class WordData:
        # Cache for WordData instances
        _cache: Dict[int, 'PerceptronAmbiguityResolver.WordData'] = {}

        def __init__(self, lemma: str, igs: Tuple[str, ...], last_ig: str):
            self.lemma = lemma
            self.igs = igs
            self._last_group = last_ig

        @classmethod
        def from_analysis(cls, sa: SingleAnalysis) -> 'PerceptronAmbiguityResolver.WordData':
            # Check cache first
            sa_id = id(sa)
            cached = cls._cache.get(sa_id)
            if cached is not None:
                return cached

            lemma = sa.item.lemma
            sec_pos: SecondaryPos = sa.item.secondary_pos
            sp: str = '' if sec_pos == SecondaryPos.None_ else sec_pos.name

            igs: List[str] = []

            for i in range(sa.group_boundaries.shape[0]):
                s: str = sa.get_group(0).lexical_form()

                if i == 0:
                    s = sp + s
                igs.append(s)

            igs_tuple = tuple(igs)
            result = cls(lemma, igs_tuple, igs[-1] if igs else '')
            cls._cache[sa_id] = result
            return result

        def last_group(self) -> str:
            return self._last_group

    class FeatureExtractor:

        feature_cache: Dict[Tuple[SingleAnalysis, ...], DefaultDict[Any, np.int32]] = dict()

        def __init__(self, use_cache: bool):
            self.use_cache = use_cache

        def extract_from_trigram(self, trigram: List[SingleAnalysis]) -> DefaultDict[Any, np.int32]:

            if self.use_cache:
                # raise ValueError(f"feature cache for FeatureExtractor has not been implemented yet!")
                cached = self.feature_cache.get(tuple(trigram))
                if cached is not None:
                    return cached

            feats = defaultdict(np.int32)

            w1: 'PerceptronAmbiguityResolver.WordData' = PerceptronAmbiguityResolver.WordData.from_analysis(
                trigram[0]
            )
            w2: 'PerceptronAmbiguityResolver.WordData' = PerceptronAmbiguityResolver.WordData.from_analysis(
                trigram[1]
            )
            w3: 'PerceptronAmbiguityResolver.WordData' = PerceptronAmbiguityResolver.WordData.from_analysis(
                trigram[2]
            )

            r1: str = w1.lemma
            r2: str = w2.lemma
            r3: str = w3.lemma

            # ig1: str = '+'.join(w1.igs)
            ig2: str = '+'.join(w2.igs)
            ig3: str = '+'.join(w3.igs)

            # r1Ig1 = f"{r1}+{ig1}"
            r2Ig2 = f"{r2}+{ig2}"
            r3Ig3 = f"{r3}+{ig3}"

            feats["2:" + r1 + ig2 + r3Ig3] += 1
            feats["3:" + r2Ig2 + "-" + r3Ig3] += 1
            feats["4:" + r3Ig3] += 1

            feats["9:" + r2 + "-" + r3] += 1
            feats["10:" + r3] += 1
            feats["10b:" + r2] += 1
            feats["10c:" + r1] += 1

            w1_last_group: str = w1.last_group()
            w2_last_group: str = w2.last_group()

            for ig in w3.igs:
                feats["15:" + w1_last_group + "-" + w2_last_group + "-" + ig] += 1
                feats["17:" + w2_last_group + ig] += 1

            for k, ig in enumerate(w3.igs):
                feats["20:" + str(k) + "-" + ig] += 1

            feats[f"22:{trigram[2].group_boundaries.shape[0]}"] += 1

            # do this outside
            # for k in feats.keys():
            #     feats[k] = np.int32(feats[k])

            if self.use_cache:
                self.feature_cache[tuple(trigram)] = feats

            return feats

    class Decoder:
        # Beam width for pruning hypotheses (limits combinatorial explosion)
        BEAM_WIDTH = 10

        def __init__(self, model: WeightLookup, extractor: 'PerceptronAmbiguityResolver.FeatureExtractor'):
            self.model = model
            self.extractor = extractor

        def best_path(self, sentence: List[WordAnalysis]) -> 'PerceptronAmbiguityResolver.DecodeResult':
            if len(sentence) == 0:
                raise ValueError("bestPath cannot be called with empty sentence.")

            # Use dict for O(1) lookup instead of O(n) linear search
            initial_hyp = PerceptronAmbiguityResolver.Hypothesis(
                PerceptronAmbiguityResolver.sentence_begin,
                PerceptronAmbiguityResolver.sentence_begin,
                previous=None,
                score=np.float32(0)
            )
            current_dict: Dict[int, 'PerceptronAmbiguityResolver.Hypothesis'] = {
                hash(initial_hyp): initial_hyp
            }

            for analysis_data in sentence:
                next_dict: Dict[int, 'PerceptronAmbiguityResolver.Hypothesis'] = {}

                analyses: List[SingleAnalysis] = list(analysis_data.analysis_results)

                if len(analyses) == 0:
                    analyses = [SingleAnalysis.unknown(analysis_data.inp)]

                for analysis in analyses:
                    for h in current_dict.values():
                        trigram: List[SingleAnalysis] = [h.prev, h.current, analysis]
                        features = self.extractor.extract_from_trigram(trigram)

                        trigram_score = np.float32(0)
                        for key in features.keys():
                            trigram_score += np.float32(self.model.get_(key) * np.float32(features.get(key)))

                        new_hyp = PerceptronAmbiguityResolver.Hypothesis(
                            h.current,
                            analysis,
                            h,
                            score=np.float32(h.score + trigram_score)
                        )

                        # O(1) dict lookup instead of O(n) linear search
                        hyp_hash = hash(new_hyp)
                        existing = next_dict.get(hyp_hash)

                        if existing is not None:
                            if new_hyp.score > existing.score:
                                next_dict[hyp_hash] = new_hyp
                        else:
                            next_dict[hyp_hash] = new_hyp

                # Beam search: keep only top-K hypotheses to prevent explosion
                if len(next_dict) > self.BEAM_WIDTH:
                    sorted_hyps = sorted(next_dict.values(), key=attrgetter('score'), reverse=True)
                    next_dict = {hash(h): h for h in sorted_hyps[:self.BEAM_WIDTH]}

                current_dict = next_dict

            # Final scoring
            for h in current_dict.values():
                trigram: List[SingleAnalysis] = [h.prev, h.current, PerceptronAmbiguityResolver.sentence_end]
                features = self.extractor.extract_from_trigram(trigram)

                trigram_score = np.float32(0)
                for key in features.keys():
                    trigram_score += np.float32(self.model.get_(key) * np.float32(features.get(key)))

                h.score += trigram_score

            best = max(current_dict.values(), key=attrgetter('score'))
            best_score = best.score
            result: List[SingleAnalysis] = []

            while best.previous is not None:
                result.append(best.current)
                best = best.previous

            return PerceptronAmbiguityResolver.DecodeResult(list(reversed(result)), best_score)

    class DecodeResult:
        def __init__(self, best_parse: List[SingleAnalysis], score: np.float32):
            self.best_parse = best_parse
            self.score = score

    class Hypothesis:
        def __init__(
                self,
                prev: SingleAnalysis,
                current: SingleAnalysis,
                previous: Optional['PerceptronAmbiguityResolver.Hypothesis'],
                score: np.float32
        ):
            self.prev = prev
            self.current = current
            self.previous = previous
            self.score = score

        def __hash__(self) -> int:
            result = hash(self.prev)
            result = 31 * result + hash(self.current)
            return result

        def __eq__(self, other):
            if self is other:
                return True
            elif isinstance(other, PerceptronAmbiguityResolver.Hypothesis):
                if self.prev != other.prev:
                    return False

                return self.current == other.current
            else:
                return False

        def __str__(self):
            return f"Hypothesis[prev='{self.prev}', current='{self.current}', score={self.score}]"
