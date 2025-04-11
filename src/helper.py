import matplotlib.pyplot as plt
from IPython.display import clear_output, display

# Enable interactive mode
plt.ion()

def plot(scores, mean_scores):
    if not scores or not mean_scores:
        return
    # Clear previous output and display the updated plot
    clear_output(wait=True)
    display(plt.gcf())
    plt.clf()

    # Set plot titles and labels
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    # Plot the scores and mean scores
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')

    # Set y-axis to start from zero
    plt.ylim(ymin=0)

    # Annotate the last score and mean score on the plot
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

    # Display the plot and pause briefly
    plt.legend()
    plt.show(block=False)
    plt.pause(0.1)
