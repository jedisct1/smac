#define smac_MACBYTES   32
#define smac_KEYBYTES   32
#define smac_NONCEBYTES 15
#define smac_BLOCKBYTES (16 * 4)

#include <stddef.h>
#include <string.h>

#include <immintrin.h>
#include <wmmintrin.h>

typedef struct {
    __m128i b0;
    __m128i b1;
    __m128i b2;
    __m128i b3;
} aes_block_t;

static inline __m128i
AES_BLOCK_XOR1(const __m128i a, const __m128i b)
{
    return _mm_xor_si128(a, b);
}

static inline aes_block_t
AES_BLOCK_XOR(const aes_block_t a, const aes_block_t b)
{
    return (aes_block_t) { _mm_xor_si128(a.b0, b.b0), _mm_xor_si128(a.b1, b.b1),
                           _mm_xor_si128(a.b2, b.b2), _mm_xor_si128(a.b3, b.b3) };
}

static inline aes_block_t
AES_BLOCK_LOAD(const uint8_t *a)
{
    return (aes_block_t) { _mm_loadu_si128((const __m128i *) (const void *) a),
                           _mm_loadu_si128((const __m128i *) (const void *) (a + 16)),
                           _mm_loadu_si128((const __m128i *) (const void *) (a + 32)),
                           _mm_loadu_si128((const __m128i *) (const void *) (a + 48)) };
}

static inline aes_block_t
AES_BLOCK_LOAD_BROADCAST(const uint8_t *a)
{
    const __m128i x = _mm_loadu_si128((const __m128i *) (const void *) a);
    return (aes_block_t) { x, x, x, x };
}

static inline aes_block_t
AES_BLOCK_LOAD_64x2(uint64_t a, uint64_t b)
{
    const __m128i t = _mm_set_epi64x((long long) a, (long long) b);
    return (aes_block_t) { t, t, t, t };
}

static inline void
AES_BLOCK_STORE(uint8_t *a, const aes_block_t b)
{
    _mm_storeu_si128((__m128i *) (void *) a, b.b0);
    _mm_storeu_si128((__m128i *) (void *) (a + 16), b.b1);
    _mm_storeu_si128((__m128i *) (void *) (a + 32), b.b2);
    _mm_storeu_si128((__m128i *) (void *) (a + 48), b.b3);
}

static inline void
AES_BLOCK_STORE1(uint8_t *a, const aes_block_t b)
{
    _mm_storeu_si128((__m128i *) (void *) a, b.b0);
}

static inline aes_block_t
AES_ENC(const aes_block_t a, const aes_block_t b)
{
    return (aes_block_t) { _mm_aesenc_si128(a.b0, b.b0), _mm_aesenc_si128(a.b1, b.b1),
                           _mm_aesenc_si128(a.b2, b.b2), _mm_aesenc_si128(a.b3, b.b3) };
}

#define PERMUTE(a)                                                                                 \
    (aes_block_t)                                                                                  \
    {                                                                                              \
        _mm_shuffle_epi8((a.b0),                                                                   \
                         _mm_setr_epi8(0, 7, 14, 11, 4, 13, 10, 1, 8, 15, 6, 3, 12, 5, 2, 9)),     \
            _mm_shuffle_epi8((a.b1),                                                               \
                             _mm_setr_epi8(0, 7, 14, 11, 4, 13, 10, 1, 8, 15, 6, 3, 12, 5, 2, 9)), \
            _mm_shuffle_epi8((a.b2),                                                               \
                             _mm_setr_epi8(0, 7, 14, 11, 4, 13, 10, 1, 8, 15, 6, 3, 12, 5, 2, 9)), \
            _mm_shuffle_epi8((a.b3),                                                               \
                             _mm_setr_epi8(0, 7, 14, 11, 4, 13, 10, 1, 8, 15, 6, 3, 12, 5, 2, 9))  \
    }

#define COMPRESS(state, m)                                                                         \
    do {                                                                                           \
        state = (smac_state) { .a1 = PERMUTE(AES_BLOCK_XOR(AES_BLOCK_XOR(state.a2, state.a3), m)), \
                               .a2 = AES_ENC(state.a1, m),                                         \
                               .a3 = AES_ENC(state.a2, m) };                                       \
    } while (0)

typedef struct smac_state {
    aes_block_t a1, a2, a3;
} smac_state;

void
smac(uint8_t tag[smac_MACBYTES], const uint8_t *ad, const size_t ad_len, const uint8_t *ct,
     const size_t ct_len, const uint8_t key[smac_KEYBYTES], const uint8_t nonce[smac_NONCEBYTES])
{
    smac_state  state, s0;
    uint8_t     nonce_patched[smac_BLOCKBYTES];
    uint8_t     pad[smac_BLOCKBYTES];
    aes_block_t m;
    size_t      i, left;

    for (i = 0; i < 4; i++) {
        memcpy(nonce_patched + i * (smac_NONCEBYTES + 1), nonce, smac_NONCEBYTES);
        nonce_patched[i * (smac_NONCEBYTES + 1) + smac_NONCEBYTES] = i | (4 << 4);
    }
    state = (smac_state) { .a1 = AES_BLOCK_LOAD_BROADCAST(key),
                           .a2 = AES_BLOCK_LOAD_BROADCAST(key + 16),
                           .a3 = AES_BLOCK_LOAD_BROADCAST(nonce_patched) };
    s0    = state;
    m     = AES_BLOCK_LOAD_64x2(0, 1);
    for (i = 0; i < 9; i++) {
        COMPRESS(state, m);
    }
    state = (smac_state) { .a1 = AES_BLOCK_XOR(state.a1, s0.a1),
                           .a2 = AES_BLOCK_XOR(state.a2, s0.a2),
                           .a3 = AES_BLOCK_XOR(state.a3, s0.a3) };

    for (i = 0; i + smac_BLOCKBYTES * 2 <= ad_len; i += smac_BLOCKBYTES * 2) {
        m = AES_BLOCK_LOAD(ad + i);
        COMPRESS(state, m);
        m = AES_BLOCK_LOAD(ad + i + smac_BLOCKBYTES);
        COMPRESS(state, m);
    }
    for (; i + smac_BLOCKBYTES <= ad_len; i += smac_BLOCKBYTES) {
        m = AES_BLOCK_LOAD(ad + i);
        COMPRESS(state, m);
    }
    left = ad_len % smac_BLOCKBYTES;
    if (left != 0) {
        memset(pad, 0, sizeof pad);
        memcpy(pad, ad, left);
        m = AES_BLOCK_LOAD(pad);
        COMPRESS(state, m);
    }

    for (i = 0; i + smac_BLOCKBYTES * 2 <= ct_len; i += smac_BLOCKBYTES * 2) {
        m = AES_BLOCK_LOAD(ct + i);
        COMPRESS(state, m);
        m = AES_BLOCK_LOAD(ct + i + smac_BLOCKBYTES);
        COMPRESS(state, m);
    }
    for (; i + smac_BLOCKBYTES <= ct_len; i += smac_BLOCKBYTES) {
        m = AES_BLOCK_LOAD(ct + i);
        COMPRESS(state, m);
    }
    left = ct_len % smac_BLOCKBYTES;
    if (left != 0) {
        memset(pad, 0, sizeof pad);
        memcpy(pad, ct, left);
        m = AES_BLOCK_LOAD(pad);
        COMPRESS(state, m);
    }

    m = AES_BLOCK_LOAD_64x2(ct_len * 8, ad_len * 8);
    COMPRESS(state, m);

    state.a1.b0 = AES_BLOCK_XOR1(AES_BLOCK_XOR1(state.a1.b1, state.a1.b2), state.a1.b3);

    s0 = state;
    m  = AES_BLOCK_LOAD_64x2(0, 1);
    for (i = 0; i < 9; i++) {
        COMPRESS(state, m);
    }
    state = (smac_state) { .a1 = AES_BLOCK_XOR(state.a1, s0.a1),
                           .a2 = AES_BLOCK_XOR(state.a2, s0.a2),
                           .a3 = AES_BLOCK_XOR(state.a3, s0.a3) };

    AES_BLOCK_STORE1(tag, state.a2);
    AES_BLOCK_STORE1(tag + 16, state.a3);
}
