# Veri kaynakları ve lisanslar

Bu proje akademik bir bitirme projesidir ve ticari kullanım amacı taşımaz. Aşağıdaki veri kaynakları yalnızca proje kapsamında kullanılmıştır.

| Kaynak | Lisans | Bu projede kullanım | Not |
|---|---|---|---|
| jyesr (Hugging Face: jyesr/pokemon-tcg-grading) | Belirtilmemiş (sayfada lisans etiketi ya da metni yok) | Not (1–10) modelinin eğitimi ve değerlendirmesi | Görsel ve not kaynağı sayfada belirtilmemiş; yalnızca akademik kullanım, veri yeniden dağıtılmaz |
| TCGdex | Site içeriği CC BY-SA 4.0; kart bilgi ve görselleri The Pokémon Company lisansında | Kart kataloğu, kart görselleri, referans fiyat (Cardmarket EUR), sentetik tespit verisi | Kart görselleri The Pokémon Company'ye aittir; ticari kullanım ve yeniden dağıtım belirtilmemiş; yalnızca akademik kullanım |
| JustTCG (ücretsiz katman) | Hizmet koşulları | Durum bazlı fiyat çarpanlarının türetilmesi | Ham fiyatlar yeniden dağıtılmaz; yalnızca türetilmiş oranlar ve kademe tabloları paylaşılır |
| Roboflow: pokemon-cards-merged v4 (card-OBB) | Public Domain | Tespit modeli test seti (set 1, 294 görsel) | Veri kümesini Roboflow kullanıcısı yüklemiştir |
| Roboflow: PokemonCardDetectCard v1 (yolo-pjxst) | CC BY 4.0 | Tespit modeli eğitim ve doğrulama (set 3, 361 + 71 görsel) | Atıf gerekli, aşağıya bakınız |
| Roboflow: pokemon_card v1 (pokemoncards-c7gbv) | CC BY 4.0 | Kullanılmadı | Set 3 ile büyük ölçüde aynı görseller içerdiği için veri silindi |
| Sentetik tespit verisi | Bu proje | Tespit modeli eğitimi (1.000 görsel) | TCGdex kart görselleri ve üretilen arka planlar kullanılmıştır |

## Yeniden dağıtım kısıtı

Aşağıdaki dosyalar JustTCG ham fiyatları içerdiği için paylaşılmamalıdır:
- `outputs/condition_ratios.csv`
- `outputs/justtcg_tcgdex_map.csv` (yalnızca eşleştirme kimlikleri, ham fiyat içermez; paylaşılabilir)

## Atıflar

PokemonCardDetectCard (yolo-pjxst), Roboflow Universe, CC BY 4.0. Kaynak: https://universe.roboflow.com/yolo-pjxst/pokemoncarddetectcard

pokemon-cards-merged (card-OBB), Roboflow Universe, Public Domain. Kaynak: https://universe.roboflow.com/pokemoncarddetect/pokemon-cards-merged

TCGdex, https://tcgdex.dev (site içeriği CC BY-SA 4.0; kart verileri ve görselleri The Pokémon Company'ye aittir)

JustTCG, https://justtcg.com (hizmet koşulları geçerlidir)
