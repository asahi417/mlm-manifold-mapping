# M3: Masked Language Model Manifold Mapping
TODO: correct prediction and see the correlation between PPL and TRUE/FALSE.
## Sample
```shell
m3-rewriter -i "AHH i'm so HAPPY." "I just found my ipod. God is sooo good to me" -n 3 -k 2 
```

```shell
m3-rewriter -f "tests/sample_sentence.txt" -n 3 -k 2 
```

```python
from m3 import Rewriter
rewriter = Rewriter()
edit = rewriter.generate("AHH i'm so HAPPY." "I just found my ipod. God is sooo good to me")
print(edit)
[[("AHH i'm so HAPPY.I just found my ipod. God is sooo good to me",
   "ahh i'm so happy!!! i just found my ipod. god is sooo good to me",
   "ahh i'm so happy!!! i just found my ipod!!! god is sooo good to me",
   "ahh i'm so happy!!! i just bought my ipod!!! god is sooo good to me",
   "ahh i'm so happy!!! i just bought an ipod!!! god is sooo good to me"),
  (54.01159819644306,
   21.694737706942327,
   15.887156689553539,
   13.448471118121727,
   11.683988133434816)]]
```

