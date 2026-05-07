from raid.utils import load_data

# Download the RAID dataset with adversarial attacks included
train_df = load_data(split="train")
test_df = load_data(split="test")
extra_df = load_data(split="extra")
breakpoint()