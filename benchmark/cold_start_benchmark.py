#!/usr/bin/env python3
"""
Cold Start Benchmark - Cache'siz Gerçek Performans Testi

Bu script cache'i devre dışı bırakarak veya bypass ederek
gerçek analiz performansını ölçer.
"""

import sys
import time
import random
import string
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_unique_words(count: int) -> list:
    """Her seferinde farklı kelimeler üret - cache miss garantisi"""

    # Türkçe kök kelimeler
    roots = [
        "ev", "araba", "kitap", "masa", "sandalye", "kalem", "defter", "okul",
        "hastane", "market", "cadde", "sokak", "park", "bahçe", "ağaç", "çiçek",
        "deniz", "göl", "nehir", "dağ", "tepe", "vadi", "orman", "tarla",
        "köy", "şehir", "ülke", "dünya", "ay", "güneş", "yıldız", "bulut",
        "yağmur", "kar", "rüzgar", "fırtına", "deprem", "sel", "yangın",
        "insan", "kadın", "erkek", "çocuk", "bebek", "genç", "yaşlı",
        "anne", "baba", "kardeş", "abla", "abi", "dede", "nine", "amca",
        "hala", "dayı", "teyze", "kuzen", "yeğen", "torun", "eş", "sevgili",
        "arkadaş", "komşu", "öğretmen", "doktor", "mühendis", "avukat",
        "polis", "asker", "şoför", "aşçı", "garson", "temizlikçi", "bekçi",
        "yemek", "içecek", "ekmek", "su", "süt", "çay", "kahve", "meyve",
        "sebze", "et", "balık", "tavuk", "pilav", "makarna", "salata",
        "gitmek", "gelmek", "almak", "vermek", "yapmak", "etmek", "olmak",
        "bilmek", "görmek", "duymak", "söylemek", "konuşmak", "okumak",
        "yazmak", "çizmek", "boyamak", "temizlemek", "yıkamak", "kurutmak",
        "açmak", "kapatmak", "başlamak", "bitirmek", "devam", "durmak",
        "koşmak", "yürümek", "oturmak", "kalkmak", "yatmak", "uyumak",
        "güzel", "çirkin", "büyük", "küçük", "uzun", "kısa", "geniş", "dar",
        "yeni", "eski", "genç", "yaşlı", "sıcak", "soğuk", "serin", "ılık"
    ]

    # Türkçe ekler
    suffixes = [
        "", "ler", "lar", "in", "ın", "un", "ün", "e", "a", "i", "ı", "u", "ü",
        "de", "da", "te", "ta", "den", "dan", "ten", "tan",
        "le", "la", "yle", "yla", "ce", "ca", "çe", "ça",
        "lik", "lık", "luk", "lük", "siz", "sız", "suz", "süz",
        "li", "lı", "lu", "lü", "ci", "cı", "cu", "cü", "çi", "çı", "çu", "çü",
        "im", "ım", "um", "üm", "in", "ın", "un", "ün",
        "imiz", "ımız", "umuz", "ümüz", "iniz", "ınız", "unuz", "ünüz",
        "ler", "lar", "leri", "ları",
        "di", "dı", "du", "dü", "ti", "tı", "tu", "tü",
        "miş", "mış", "muş", "müş",
        "yor", "iyor", "ıyor", "uyor", "üyor",
        "ecek", "acak", "ir", "ır", "ur", "ür", "er", "ar",
        "meli", "malı", "ebil", "abil",
        "ken", "ince", "ınca", "unca", "ünce",
        "dik", "dık", "duk", "dük", "tik", "tık", "tuk", "tük",
        "dikçe", "dıkça", "dukça", "dükçe"
    ]

    words = []
    used = set()

    while len(words) < count:
        root = random.choice(roots)
        # 1-3 ek ekle
        num_suffixes = random.randint(1, 3)
        word = root
        for _ in range(num_suffixes):
            suffix = random.choice(suffixes)
            word += suffix

        # Benzersizlik için rastgele sayı ekle
        unique_word = f"{word}{random.randint(1000, 9999)}"

        if unique_word not in used:
            used.add(unique_word)
            words.append(unique_word)

    return words


def generate_unique_sentences(count: int) -> list:
    """Benzersiz cümleler üret"""

    templates = [
        "{} {} {}.",
        "{} ve {} {} {}.",
        "{} {} {} {} {}.",
        "Bu {} çok {}.",
        "{} {} ile {} {}.",
        "Dün {} {} {}.",
        "Yarın {} {} olacak.",
        "{} {} için {} gerekiyor.",
        "Her {} {} {} ister.",
        "{} {} kadar {} değil."
    ]

    words = generate_unique_words(count * 10)
    sentences = []

    for i in range(count):
        template = random.choice(templates)
        num_slots = template.count("{}")
        slot_words = [words[i * 10 + j] for j in range(num_slots)]
        sentence = template.format(*slot_words)
        sentences.append(sentence)

    return sentences


def benchmark_without_cache(limit: int = 10000):
    """Cache olmadan benchmark çalıştır"""

    print("=" * 60)
    print("  COLD START BENCHMARK - Cache'siz Gerçek Performans")
    print("=" * 60)
    print()

    # 1. Morphology'yi yükle
    print("[1/5] TurkishMorphology yükleniyor...")
    start = time.time()

    from zemberek.morphology import TurkishMorphology
    morphology = TurkishMorphology.create_with_defaults()

    load_time = time.time() - start
    print(f"      Yükleme süresi: {load_time*1000:.0f}ms")
    print()

    # 2. Cache'i temizle
    print("[2/5] Cache temizleniyor...")
    try:
        from zemberek.cache import CacheManager
        cache_manager = CacheManager.get_instance()
        # L2 cache'i temizle (L0 precomputed, temizlenemez)
        cache_manager._l2._java_hash.clear()
        cache_manager._l2._str_hash.clear()
        cache_manager.reset_stats()
        # analyze() lru_cache'i temizle
        morphology.analyze.cache_clear()
        print("      L2 cache ve lru_cache temizlendi")
    except Exception as e:
        print(f"      Cache temizleme hatası: {e}")
    print()

    # 3. Benzersiz kelimeler üret (cache miss garantisi)
    print(f"[3/5] {limit} benzersiz kelime üretiliyor...")
    unique_words = generate_unique_words(limit)
    print(f"      {len(unique_words)} benzersiz kelime üretildi")
    print()

    # 4. Kelime analizi benchmark (cache bypass)
    print("[4/5] Kelime analizi benchmark (cache'siz)...")
    print()

    # Warm-up (sadece 10 kelime, sonra cache temizle)
    for word in unique_words[:10]:
        morphology.analyze(word)

    # Cache'i tekrar temizle
    try:
        cache_manager._l2._java_hash.clear()
        cache_manager._l2._str_hash.clear()
        cache_manager.reset_stats()
        morphology.analyze.cache_clear()
    except:
        pass

    # Gerçek benchmark
    times = []
    analyzed_count = 0

    batch_size = 1000
    total_batches = limit // batch_size

    print(f"      {limit} kelime analiz ediliyor ({total_batches} batch x {batch_size})...")
    print()

    overall_start = time.time()

    for batch_idx in range(total_batches):
        batch_start = time.time()
        batch_words = unique_words[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        for word in batch_words:
            word_start = time.time()
            result = morphology.analyze(word)
            word_time = time.time() - word_start
            times.append(word_time)
            analyzed_count += 1

        batch_time = time.time() - batch_start
        throughput = batch_size / batch_time

        # Her batch sonrası cache temizle (cold start simülasyonu)
        try:
            cache_manager._l2._java_hash.clear()
            cache_manager._l2._str_hash.clear()
            morphology.analyze.cache_clear()
        except:
            pass

        print(f"      Batch {batch_idx + 1}/{total_batches}: {throughput:.1f} kelime/sn")

    overall_time = time.time() - overall_start

    # 5. Sonuçları hesapla
    print()
    print("[5/5] Sonuçlar hesaplanıyor...")
    print()

    times_sorted = sorted(times)
    mean_time = sum(times) / len(times) * 1000
    p50_time = times_sorted[int(len(times) * 0.50)] * 1000
    p90_time = times_sorted[int(len(times) * 0.90)] * 1000
    p95_time = times_sorted[int(len(times) * 0.95)] * 1000
    p99_time = times_sorted[int(len(times) * 0.99)] * 1000
    min_time = min(times) * 1000
    max_time = max(times) * 1000

    throughput = analyzed_count / overall_time

    # Cache durumunu kontrol et
    try:
        stats = cache_manager.stats()
        cache_hits = stats.get('l0_hits', 0) + stats.get('l2_hits', 0)
        cache_misses = stats.get('misses', 0)
        hit_rate = (cache_hits / (cache_hits + cache_misses) * 100) if (cache_hits + cache_misses) > 0 else 0
    except:
        hit_rate = 0
        cache_hits = 0
        cache_misses = 0

    print("=" * 60)
    print("  COLD START BENCHMARK SONUÇLARI")
    print("=" * 60)
    print()
    print(f"  Test Konfigürasyonu:")
    print(f"    Kelime sayısı:     {limit:,}")
    print(f"    Benzersiz kelime:  %100 (cache miss garantisi)")
    print(f"    Cache:             Devre dışı (her batch sonrası temizlendi)")
    print()
    print(f"  Performans Metrikleri:")
    print(f"    Toplam süre:       {overall_time:.2f}s")
    print(f"    Throughput:        {throughput:.1f} kelime/sn")
    print(f"    Ortalama:          {mean_time:.3f}ms/kelime")
    print(f"    Median (p50):      {p50_time:.3f}ms")
    print(f"    p90:               {p90_time:.3f}ms")
    print(f"    p95:               {p95_time:.3f}ms")
    print(f"    p99:               {p99_time:.3f}ms")
    print(f"    Min:               {min_time:.3f}ms")
    print(f"    Max:               {max_time:.3f}ms")
    print()
    print(f"  Cache Durumu (doğrulama):")
    print(f"    Hit rate:          {hit_rate:.1f}% (beklenen: ~0%)")
    print()
    print("=" * 60)

    # Cümle analizi de test edelim
    print()
    print("=" * 60)
    print("  CÜMLE ANALİZİ (Cold Start)")
    print("=" * 60)
    print()

    sentence_count = min(1000, limit // 10)
    print(f"[1/2] {sentence_count} benzersiz cümle üretiliyor...")
    sentences = generate_unique_sentences(sentence_count)
    print()

    # Cache temizle
    try:
        cache_manager._l2._java_hash.clear()
        cache_manager._l2._str_hash.clear()
        cache_manager.reset_stats()
        morphology.analyze.cache_clear()
    except:
        pass

    print(f"[2/2] Cümle analizi (disambiguate)...")

    sentence_times = []
    sentence_start = time.time()

    for i, sentence in enumerate(sentences):
        s_start = time.time()
        result = morphology.analyze_and_disambiguate(sentence)
        s_time = time.time() - s_start
        sentence_times.append(s_time)

        if (i + 1) % 100 == 0:
            # Her 100 cümle sonrası cache temizle
            try:
                cache_manager._l2._java_hash.clear()
                cache_manager._l2._str_hash.clear()
                morphology.analyze.cache_clear()
            except:
                pass
            batch_throughput = 100 / sum(sentence_times[-100:])
            print(f"      {i+1}/{sentence_count}: {batch_throughput:.1f} cümle/sn")

    sentence_total = time.time() - sentence_start
    sentence_throughput = sentence_count / sentence_total

    sentence_times_sorted = sorted(sentence_times)
    s_mean = sum(sentence_times) / len(sentence_times) * 1000
    s_p50 = sentence_times_sorted[int(len(sentence_times) * 0.50)] * 1000
    s_p99 = sentence_times_sorted[int(len(sentence_times) * 0.99)] * 1000

    print()
    print(f"  Cümle Analizi Sonuçları:")
    print(f"    Cümle sayısı:      {sentence_count}")
    print(f"    Toplam süre:       {sentence_total:.2f}s")
    print(f"    Throughput:        {sentence_throughput:.1f} cümle/sn")
    print(f"    Ortalama:          {s_mean:.2f}ms/cümle")
    print(f"    Median (p50):      {s_p50:.2f}ms")
    print(f"    p99:               {s_p99:.2f}ms")
    print()
    print("=" * 60)
    print("  BENCHMARK TAMAMLANDI")
    print("=" * 60)

    return {
        "word_analysis": {
            "count": limit,
            "throughput": throughput,
            "mean_ms": mean_time,
            "p50_ms": p50_time,
            "p99_ms": p99_time
        },
        "sentence_analysis": {
            "count": sentence_count,
            "throughput": sentence_throughput,
            "mean_ms": s_mean,
            "p50_ms": s_p50,
            "p99_ms": s_p99
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cold Start Benchmark")
    parser.add_argument("--limit", type=int, default=10000, help="Kelime sayısı")
    args = parser.parse_args()

    results = benchmark_without_cache(args.limit)
